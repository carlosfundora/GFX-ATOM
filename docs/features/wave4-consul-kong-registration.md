# Wave-4 Feature: Consul + Kong Registration Lane

## Goal

Register the new gfxATOM engine/API surfaces under governance-compliant service names and port blocks, and wire API ingress through Kong.

## Consul service definitions

Updated service specs:

- `/home/local/ai/consul/services/inference.json`
  - `gfxatom-rust` on `40013`
- `/home/local/ai/consul/services/api.json`
  - `api-gfxatom` on `9314`

Both entries use:

- domain-aligned names (`400xx` inference, `93xx` API)
- lifecycle tags (`on_demand`)
- `meta.project=ai`
- `/health` checks

## Registry resolution layer

Refreshed generated port map via:

- `python3 /home/local/ai/projects/scripts/generate_consul_env.py`

New variables:

- `PORT_GFXATOM_RUST=40013`
- `PORT_API_GFXATOM=9314`

## Kong route integration

Updated declarative gateway config:

- `ENCOM/servers/gateway/kong.yaml`
  - service: `api-gfxatom`
  - route path: `/api/gfxatom`
  - upstream: `http://host.docker.internal:9314`
