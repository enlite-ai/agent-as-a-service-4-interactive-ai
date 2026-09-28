# A3S architecture

Agent-as-a-Service answers one question — *"given the state an operator is
looking at, what could they do, and how does each option play out?"* — and it
answers it the same way for every environment. That symmetry is the whole
design: the machinery for **projecting** a decision forward is generic, and
everything that makes a domain specific is pushed behind two interfaces.

The codebase therefore splits in two, with the dependency running one way only:

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 25, "rankSpacing": 45, "curve": "basis"}}}%%
flowchart TB
    subgraph transport["api/ · HTTP transport"]
        direction LR
        views["<b>views.py</b><br/><i>routes, request/response</i>"]
        utils["<b>utils.py</b><br/><i>reads A3S_USE_CASES</i>"]
    end

    subgraph impls["integrations/ · one package per environment"]
        direction LR
        pg["<b>powergrid/</b><br/>Grid2Op<br/><i>implemented</i>"]
        other["<b>your_env/</b><br/><i>same three pieces</i>"]
    end

    subgraph core["a3s_core/ · reusable, environment-agnostic"]
        direction LR
        iface["<b>interface.py</b><br/>Environment · Agent<br/><i>the contract</i>"]
        uc["<b>use_case.py</b><br/>UseCase<br/><i>the plugin surface</i>"]
        service["<b>service.py</b><br/>RecommendationService<br/><i>the projection engine</i>"]
        ser["<b>serialization.py</b><br/><i>opaque state envelope</i>"]
        reg["<b>registry.py</b><br/><i>loads plugins by import path</i>"]
        iface ~~~ uc ~~~ service ~~~ ser ~~~ reg
    end

    transport -->|resolves by name| core
    impls -->|implement| core
    core -.->|"registry.py imports at startup, by configured path only"| impls

    classDef coreStyle fill:#e8f0fe,stroke:#4a6fa5,stroke-width:2px
    classDef implStyle fill:#eef7ee,stroke:#5a8f5a,stroke-width:2px
    classDef transportStyle fill:#faf0e8,stroke:#a5804a,stroke-width:2px
    class core,service,iface,uc,reg,ser coreStyle
    class impls,pg,other implStyle
    class transport,views,utils transportStyle
```

`a3s_core/` names no environment and imports nothing from `integrations/`. The
dotted arrow is the single, deliberate exception: `registry.py` imports whatever
`A3S_USE_CASES` names, by string, at startup — so the core reaches an
integration through configuration rather than through a source dependency.

## The contract

Three abstractions carry everything domain-specific. An integration supplies
concrete versions; the core is written against nothing else.

```mermaid
classDiagram
    direction LR

    class Environment {
        <<abstract>>
        +prepare(state) Observation
        +step(action) (Observation, done)
        +fork() Environment
        +kpis(observation) KPIs
        +timestep(observation) int
        +format_recommendation(...) A3SRecommendation
    }

    class Agent {
        <<abstract>>
        +agent_type: str
        +propose(observation, n) list~Action~
        +act(observation) Action
    }

    class UseCase {
        <<abstract>>
        +name: str
        +build_service() AgentAsAService
        +resolve_environment_state(context) Envelope
    }

    class EnvironmentUseCase {
        <<abstract>>
        +build_environment() Environment
        +build_agent(env) Agent
        +validate_envelope(envelope)
        +adapt_legacy_context(context)
    }

    class PowerGridEnvironment {
        Grid2Op with LightSim2Grid
        reconstructs by action replay
        forks by deep copy
        KPIs are rho and line loading
    }

    class PowerGridAgent {
        XD_silly_repo planner
        simulates and ranks greedily
    }

    class ExpertPowerGridAgent {
        T2.1 PPO policy
        ranks by action logits
    }

    class PowerGridUseCase {
        name is PowerGrid
        engine set by A3S_POWERGRID_AGENT
    }

    UseCase <|-- EnvironmentUseCase
    EnvironmentUseCase <|-- PowerGridUseCase
    Environment <|.. PowerGridEnvironment
    Agent <|.. PowerGridAgent
    Agent <|.. ExpertPowerGridAgent

    PowerGridUseCase ..> PowerGridEnvironment : build_environment()
    PowerGridUseCase ..> PowerGridAgent : build_agent() if xd
    PowerGridUseCase ..> ExpertPowerGridAgent : build_agent() if t2.1
```

Why these three and not fewer:

| Abstraction | Answers | Why it can't live in the core |
| --- | --- | --- |
| `Environment` | *How do I reconstruct, advance and branch a world, and what is a KPI in it?* | Every simulator reconstructs differently, and picks a different cheap branching primitive — some are expensive to step but cheap to copy, others the reverse. |
| `Agent` | *Which actions are worth showing, and what would the policy do next?* | This is the recommendation engine — a planner, a trained policy, a rule set. Interchangeable even within one environment. |
| `UseCase` | *Which of the above, under what name, reading which payload shape?* | Wiring and payload dialects are deployment facts, so the HTTP layer stays free of them. |

## How a request flows

The engine never simulates or copies directly. It advances the world through
`step` and gets independent branches through `fork`, leaving each environment
free to implement those the cheap way.

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant V as api/views
    participant U as PowerGridUseCase
    participant S as RecommendationService<br/>(core)
    participant E as PowerGridEnvironment
    participant A as Agent<br/>(engine)

    C->>V: POST /api/v1/recommendation<br/>{context, options}
    V->>U: resolve_environment_state(context)
    Note over U: envelope → validate_grid2op_envelope<br/>bare payload → wrap as legacy
    U-->>V: SerializedEnvironmentState
    V->>S: get_recommendations(request)

    Note over S: acquire lock — one rollout at a time,<br/>the prepared env is shared and mutable
    S->>E: prepare(state)
    E-->>S: root observation (or None → no recos)

    S->>A: propose(root_obs, max_recommendations)
    A-->>S: k alternative first actions

    loop per alternative (branch_index)
        S->>E: fork()
        E-->>S: independent branch
        loop n_steps + 1 timesteps
            S->>E: step(action)
            E-->>S: observation, done
            S->>E: kpis(obs) · timestep(obs)
            S->>E: format_recommendation(...)
            E-->>S: A3SRecommendation
            alt done
                Note over S: branch ends early
            else continue
                S->>A: act(observation)
                A-->>S: next action
            end
        end
    end

    S-->>V: list[A3SRecommendation]
    V-->>C: 200 · normalized JSON
```

### What the fan-out produces

`max_recommendations` sets the width, `n_steps` the depth. Each branch applies
one alternative, then follows the policy — so a branch spans `n_steps + 1`
timesteps, the first being the operator-facing option and the rest the
projection of what the agent would do after it.

```mermaid
flowchart LR
    root(["root observation<br/><i>prepare(state)</i>"])

    root -->|"propose() → option A"| a1["step 1<br/><i>the recommendation</i>"]
    root -->|"option B"| b1["step 1"]
    root -->|"option C"| c1["step 1"]

    a1 -->|"act()"| a2["step 2"] -->|"act()"| a3["step 3"]
    b1 -->|"act()"| b2["step 2"] -->|"act()"| b3["step 3"]
    c1 -->|"act()"| c2["step 2"] --> cx(["done<br/><i>terminal KPIs</i>"])

    classDef reco fill:#e8f0fe,stroke:#4a6fa5,stroke-width:2px
    classDef proj fill:#f5f5f5,stroke:#999
    classDef term fill:#fdecea,stroke:#c0392b
    class a1,b1,c1 reco
    class a2,a3,b2,b3,c2 proj
    class cx term
```

Every emitted recommendation carries `branch_index` (which option) and `step`
(how far along that option), plus `env_timestep` — the environment's own
absolute clock, a different clock from `step`, so a projection can be placed on
a real timeline. A `done` step reports a *terminal* state, whose KPIs may be
degenerate and must not be read as ordinary values.

## What PowerGrid actually adds

The Grid2Op integration is the reference implementation of the contract above.
Everything in it is domain work; none of it is projection logic.

```mermaid
flowchart TB
    subgraph pg["integrations/powergrid/"]
        direction TB
        usecase["<b>use_case.py</b> · PowerGridUseCase<br/><i>the only wiring — names env + engine</i>"]

        subgraph env["Environment implementation"]
            direction LR
            environment["environment.py<br/><i>reconstruct · step · fork</i>"]
            serial["serialization.py<br/><i>Grid2Op envelope, replay decode</i>"]
            kpi["kpi.py<br/><i>rho, line loading</i>"]
            fmt["formatting.py<br/><i>native action → prose</i>"]
        end

        subgraph engines["Agent implementations"]
            direction LR
            xd["agent_implementations/xd_agent.py<br/><i>XD planner — greedy,<br/>simulate-and-rank</i>"]
            t21["agent_implementations/t2_1_agent.py<br/><i>T2.1 PPO policy</i>"]
            sim["simulation.py<br/><i>shared what-if helpers</i>"]
        end
    end

    usecase --> env
    usecase --> engines
    environment --> serial
    environment --> kpi
    environment --> fmt
    xd --> sim
    t21 --> sim

    classDef wiring fill:#e8f0fe,stroke:#4a6fa5,stroke-width:2px
    classDef domain fill:#eef7ee,stroke:#5a8f5a
    class usecase wiring
    class environment,serial,kpi,fmt,xd,t21,sim domain
```

`PowerGridUseCase` is the whole of the integration's wiring, and it is
deliberately thin — a name, an environment, and a choice of engine:

```python
class PowerGridUseCase(EnvironmentUseCase):
    @property
    def name(self):
        return "PowerGrid"

    def build_environment(self):
        return PowerGridEnvironment()

    def build_agent(self, environment):
        # "xd" (default) or "t2.1", chosen by A3S_POWERGRID_AGENT
        return RECOMMENDATION_ENGINES[engine_name](environment)
```

The two engines are swapped by `A3S_POWERGRID_AGENT` and are otherwise
identical to the core — same environment, same projection, same response shape
— which is what makes them comparable on identical requests.

One thing worth naming: `adapt_legacy_context` wraps a bare Grid2Op observation
into the standard envelope, keeping the service a drop-in replacement for the
older T2.1_deep_expert API. That dialect detail never reaches the core or the
HTTP layer.

## Adding an environment

```mermaid
flowchart LR
    s1["<b>1</b><br/>Environment subclass<br/><i>reconstruct, step, fork,<br/>KPIs, clock, render</i>"]
    s2["<b>2</b><br/>Agent subclass<br/><i>propose + act</i>"]
    s3["<b>3</b><br/>EnvironmentUseCase<br/><i>name the two above</i>"]
    s4["<b>4</b><br/>Set A3S_USE_CASES<br/><i>module:Class</i>"]
    done(["Served — no change in<br/>a3s_core/ or api/"])

    s1 --> s3
    s2 --> s3
    s3 --> s4 --> done

    classDef step fill:#eef7ee,stroke:#5a8f5a,stroke-width:2px
    classDef result fill:#e8f0fe,stroke:#4a6fa5,stroke-width:2px
    class s1,s2,s3,s4 step
    class done result
```

`A3S_USE_CASES` takes a comma-separated list, so one deployment can serve
several environments at once; callers select one with the `use_case` query
parameter, and `A3S_DEFAULT_USE_CASE` names the one that serves requests
omitting it (otherwise the first in the list). A malformed or unresolvable spec
fails at startup rather than silently serving nothing, so a typo stays
distinguishable from a deliberate configuration.

## Environment state and reconstruction fidelity

The environment is rebuilt from the serialized state the simulator publishes.
How close it gets depends on what the producer sends:

* **Exact** — the state carries the env's identity and its full `replay_actions`
  history. A3S seeds an identical env and replays that history, which also
  restores the hidden state an observation can't express (cooldowns, opponent
  budget, accumulated redispatch). This is what makes a multi-step projection
  trustworthy. The replayed prefix is cached, so a follow-up request from the
  same producer only steps the new actions.
* **Approximate** — no history, so the env is fast-forwarded through the
  chronics to the incoming timestep and the observation is overlaid. The hidden
  state is lost, so a projection drifts from what the simulator would reach.

For the exact path the producer and A3S must agree on scenario and seed (both
read them from their `CONFIG_*.toml`), on the basename of the Grid2Op env data
directory (`env_ICAPS_input_data_test`), and on the Grid2Op / LightSim2Grid
versions.
