# Project Ironwood — Briefing

- Team: 6 engineers (4 backend, 1 frontend, 1 SRE), led by Dana Okoro.
- Mission: a real-time inventory-sync service for warehouse robotics.
- Language/stack: Rust backend services; TypeScript admin console; Postgres 15
  as the system of record; Redis for ephemeral robot-position cache.
- Service comms: gRPC between internal services; REST+JSON at the public edge.
- Deploy: Kubernetes on AWS (eu-west-1), one prod cluster, blue/green rollouts.
- Decision (ADR-007): chose gRPC over REST internally for streaming + typed
  contracts; accepted heavier tooling cost.
- Decision (ADR-012): Postgres over DynamoDB — relational invariants on
  inventory counts mattered more than infinite scale.
- Constraint: hard p99 latency budget of 50ms for position lookups.
- Constraint: no PII may leave eu-west-1 (GDPR data-residency).
- Known risk: the Redis cache is a single point of failure; HA is backlogged.
