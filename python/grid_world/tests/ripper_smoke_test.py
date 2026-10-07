"""
Created on Sun Oct  4 21:07:42 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

# test_bindings.py
import ripper

print("=" * 50)
print("1) Build a world from a simple layout")
print("=" * 50)

layout = (
    "##########\n"
    "#P.......#\n"
    "#..E...T.#\n"
    "#........#\n"
    "#.......G#\n"
    "##########"
)

config = ripper.WorldConfig(max_tick=50, player_perception_range=3)
world = ripper.World(layout, config=config, seed=42)

print(f"width={world.width} height={world.height} tick={world.tick} done={world.done}")
print(world.render_world())

print("\n" + "=" * 50)
print("2) Step the player around, inspect the TimeStep")
print("=" * 50)

for action in [ripper.Action.Right, ripper.Action.Right, ripper.Action.Down]:
    ts = world.step({0: action})
    print(f"action={action} tick={world.tick} terminated={ts.terminated} truncated={ts.truncated} done={ts.done()}")

print("\nPlayer's own view:")
print(world.render_entity_view(0))

print("\nFull observation sanity check:")
obs = ts.observation
print(f"player_id={obs.player_id} enemy_ids={obs.enemy_ids} n_entities={len(obs.entities)}")
for ent in obs.entities:
    print(f"  id={ent.id} pos={ent.position} status={ent.status} grid_rows={len(ent.grid)}")

print("\n" + "=" * 50)
print("3) Reset, with and without reposition")
print("=" * 50)

ts = world.reset()
print("plain reset -> tick:", world.tick, "done:", world.done)

ts = world.reset(reposition=True)
print("reset(reposition=True) -> tick:", world.tick, "done:", world.done)
print(world.render_world())

print("\n" + "=" * 50)
print("4) Procedural generation (validity-guaranteed)")
print("=" * 50)

gen_config = ripper.GenerationConfig(width=10, height=10, num_enemies=2, num_traps=1, seed=7)
generated = ripper.generate_valid_layout(gen_config)
print(generated if generated else "generation failed after max attempts")

print("\n" + "=" * 50)
print("5) Mutation (validity-guaranteed)")
print("=" * 50)

mut_config = ripper.MutationConfig(seed=7)
mutated = ripper.mutate_valid_layout(generated, mut_config)
print(mutated if mutated else "mutation failed after max attempts")

print("\nAll smoke checks ran without exceptions.")