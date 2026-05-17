package sensor

import (
	"context"
	"sort"
	"sync"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// StandardSensorChain is the reference in-memory SensorChain. Sufficient
// for tests, short-lived processes, and as a building block under a
// durable wrapper.
type StandardSensorChain struct {
	mu          sync.Mutex
	sensors     []sensorEntry
	history     []historyRecord
	sessions    map[sporecore.SessionID]struct{}
	nowOverride *Timestamp
}

type sensorEntry struct {
	config SensorConfig
	sensor Sensor
}

type historyRecord struct {
	sensorID  SensorID
	sessionID sporecore.SessionID
	outcome   SensorOutcome
	firedAt   Timestamp
}

// NewStandardSensorChain constructs an empty chain.
func NewStandardSensorChain() *StandardSensorChain {
	return &StandardSensorChain{
		sessions: make(map[sporecore.SessionID]struct{}),
	}
}

// SetNow pins the "now" timestamp returned by nowTimestamp. Tests use this
// for deterministic results.
func (c *StandardSensorChain) SetNow(now Timestamp) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.nowOverride = &now
}

func (c *StandardSensorChain) nowTimestamp() Timestamp {
	if c.nowOverride != nil {
		return *c.nowOverride
	}
	return Timestamp(time.Now().UTC().Format(time.RFC3339))
}

// inferentialGateOpen decides whether an Inferential sensor should run for
// this input. Computational sensors are always permitted regardless of
// gating fields.
func inferentialGateOpen(cfg SensorConfig, input *SensorInput) bool {
	if cfg.Kind == SensorKindComputational {
		return true
	}
	if len(cfg.RunOnPhases) > 0 {
		if input.Phase == nil {
			return false
		}
		match := false
		for _, p := range cfg.RunOnPhases {
			if p == *input.Phase {
				match = true
				break
			}
		}
		if !match {
			return false
		}
	}
	if cfg.RunEveryNTurns != nil {
		n := *cfg.RunEveryNTurns
		if n == 0 {
			return false
		}
		// Fire when turn_number is a multiple of n. Missing turn_number
		// means we cannot gate — default to firing so the caller pays
		// the cost rather than silently dropping inferential evidence.
		if input.TurnNumber != nil {
			if *input.TurnNumber%n != 0 {
				return false
			}
		}
	}
	return true
}

// Register validates and inserts a sensor.
func (c *StandardSensorChain) Register(_ context.Context, s Sensor) error {
	cfg := s.Config()
	if len(cfg.Triggers) == 0 {
		return &SensorError{
			Kind:   ErrKindValidationFailed,
			Reason: "sensor must declare at least one trigger",
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, e := range c.sensors {
		if e.config.ID == cfg.ID {
			return &SensorError{Kind: ErrKindAlreadyRegistered, SensorID: cfg.ID}
		}
	}
	c.sensors = append(c.sensors, sensorEntry{config: cfg, sensor: s})
	return nil
}

// Fire runs every matching sensor and returns every result.
func (c *StandardSensorChain) Fire(ctx context.Context, trigger SensorTrigger, input *SensorInput) []SensorResult {
	// Snapshot eligible sensors under the lock, then evaluate outside the
	// lock so Evaluate cannot deadlock by re-entering the chain.
	type candidate struct {
		id     SensorID
		sensor Sensor
	}
	c.mu.Lock()
	c.sessions[input.SessionID] = struct{}{}
	candidates := make([]candidate, 0, len(c.sensors))
	for _, e := range c.sensors {
		matched := false
		for _, t := range e.config.Triggers {
			if t.Matches(trigger) {
				matched = true
				break
			}
		}
		if !matched {
			continue
		}
		if !inferentialGateOpen(e.config, input) {
			continue
		}
		candidates = append(candidates, candidate{id: e.config.ID, sensor: e.sensor})
	}
	c.mu.Unlock()

	results := make([]SensorResult, 0, len(candidates))
	for _, cand := range candidates {
		r := cand.sensor.Evaluate(ctx, input)
		c.mu.Lock()
		c.history = append(c.history, historyRecord{
			sensorID:  cand.id,
			sessionID: input.SessionID,
			outcome:   r.Outcome,
			firedAt:   r.FiredAt,
		})
		c.mu.Unlock()
		results = append(results, r)
	}
	return results
}

// Stats aggregates the firing history per sensor.
func (c *StandardSensorChain) Stats(_ context.Context, since *Timestamp) []SensorStats {
	c.mu.Lock()
	defer c.mu.Unlock()

	type agg struct {
		total uint32
		pass  uint32
		warn  uint32
		halt  uint32
		last  *Timestamp
	}
	bySensor := make(map[SensorID]*agg, len(c.sensors))
	for _, e := range c.sensors {
		bySensor[e.config.ID] = &agg{}
	}
	for _, rec := range c.history {
		if since != nil && string(rec.firedAt) < string(*since) {
			continue
		}
		a, ok := bySensor[rec.sensorID]
		if !ok {
			a = &agg{}
			bySensor[rec.sensorID] = a
		}
		a.total++
		switch rec.outcome {
		case OutcomePass:
			a.pass++
		case OutcomeWarn:
			a.warn++
		case OutcomeHalt:
			a.halt++
		}
		ts := rec.firedAt
		a.last = &ts
	}

	sessionsTotal := float32(len(c.sessions))
	out := make([]SensorStats, 0, len(bySensor))
	for id, a := range bySensor {
		var cfg *SensorConfig
		for i := range c.sensors {
			if c.sensors[i].config.ID == id {
				cfg = &c.sensors[i].config
				break
			}
		}
		var fireRate float32
		if sessionsTotal > 0 {
			fireRate = float32(a.total) / sessionsTotal
			if fireRate > 1.0 {
				fireRate = 1.0
			}
			if fireRate < 0.0 {
				fireRate = 0.0
			}
		}
		lowSignal := false
		if cfg != nil {
			if fireRate > cfg.LowSignalThreshold.AlwaysFiredRate {
				lowSignal = true
			} else if a.total == 0 && uint32(len(c.sessions)) >= cfg.LowSignalThreshold.NeverFiredAfterNSessions {
				lowSignal = true
			}
		}
		out = append(out, SensorStats{
			SensorID:      id,
			TotalFires:    a.total,
			WarnCount:     a.warn,
			HaltCount:     a.halt,
			PassCount:     a.pass,
			FireRate:      fireRate,
			LastFired:     a.last,
			LowSignalFlag: lowSignal,
		})
	}
	sort.Slice(out, func(i, j int) bool { return string(out[i].SensorID) < string(out[j].SensorID) })
	return out
}

// SignalQualityReport flags NeverFired / AlwaysFiring sensors. Returns
// empty when the chain has observed fewer than minSessions sessions.
func (c *StandardSensorChain) SignalQualityReport(_ context.Context, minSessions uint32) []SensorSignalFlag {
	c.mu.Lock()
	defer c.mu.Unlock()

	sessionsObserved := uint32(len(c.sessions))
	out := make([]SensorSignalFlag, 0)
	if sessionsObserved < minSessions {
		return out
	}

	for _, e := range c.sensors {
		var total uint32
		for _, rec := range c.history {
			if rec.sensorID == e.config.ID {
				total++
			}
		}
		var fireRate float32
		if sessionsObserved > 0 {
			fireRate = float32(total) / float32(sessionsObserved)
			if fireRate > 1.0 {
				fireRate = 1.0
			}
		}
		switch {
		case total == 0 && sessionsObserved >= e.config.LowSignalThreshold.NeverFiredAfterNSessions:
			out = append(out, NewFlagNeverFired(e.config.ID, sessionsObserved))
		case fireRate > e.config.LowSignalThreshold.AlwaysFiredRate:
			out = append(out, NewFlagAlwaysFiring(e.config.ID, fireRate))
		}
	}
	// Deterministic order: NeverFired group first (by id), then AlwaysFiring (by id).
	sort.SliceStable(out, func(i, j int) bool {
		ki := func(f SensorSignalFlag) (int, string) {
			if f.Kind == FlagKindNeverFired {
				return 0, string(f.SensorID)
			}
			return 1, string(f.SensorID)
		}
		gi, si := ki(out[i])
		gj, sj := ki(out[j])
		if gi != gj {
			return gi < gj
		}
		return si < sj
	})
	return out
}

// Compile-time interface check.
var _ SensorChain = (*StandardSensorChain)(nil)
