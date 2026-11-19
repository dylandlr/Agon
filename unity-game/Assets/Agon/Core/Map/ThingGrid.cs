using System.Collections.Generic;
using UnityEngine;

namespace Agon.Core
{
    /// <summary>What a pawn can go and use. Enough kinds to drive the needs; not an item system.</summary>
    public enum ThingKind : byte
    {
        Bed = 0,
        FoodStore = 1,
        WaterSource = 2,
        Hearth = 3
    }

    public class Thing
    {
        public readonly ThingKind Kind;
        public Vector2Int Position;

        /// <summary>How much is left in it, for the kinds that hold something. -1 is inexhaustible.</summary>
        public float Stock;

        /// <summary>The pawn that has claimed this, so two do not walk to the same bed.</summary>
        public Pawn ReservedBy;

        public Thing(ThingKind kind, Vector2Int position, float stock = -1f)
        {
            Kind = kind;
            Position = position;
            Stock = stock;
        }

        public bool Depleted => Stock >= 0f && Stock <= 0f;
    }

    /// <summary>
    /// The usable things on a map. Reservation lives here rather than on the pawn,
    /// because the question a job giver asks is "is anything free", which is a
    /// property of the thing.
    /// </summary>
    public class ThingGrid
    {
        private readonly Map _map;
        private readonly List<Thing> _things = new List<Thing>();

        public ThingGrid(Map map) { _map = map; }

        public IReadOnlyList<Thing> All => _things;

        public Thing Add(ThingKind kind, Vector2Int position, float stock = -1f)
        {
            var thing = new Thing(kind, position, stock);
            _things.Add(thing);
            return thing;
        }

        public void Remove(Thing thing) => _things.Remove(thing);

        /// <summary>
        /// The nearest usable thing of a kind, by octile distance — the same measure the
        /// pathfinder's heuristic uses, so the nearest candidate is also the cheapest to
        /// reach over open ground. Anything depleted, or claimed by another pawn, is
        /// skipped.
        /// </summary>
        public Thing NearestUsable(ThingKind kind, Vector2Int from, Pawn forPawn)
        {
            Thing best = null;
            float bestDist = float.MaxValue;

            for (int i = 0; i < _things.Count; i++)
            {
                Thing t = _things[i];

                if (t.Kind != kind || t.Depleted)
                    continue;

                if (t.ReservedBy != null && t.ReservedBy != forPawn)
                    continue;

                float d = PathFinder.Heuristic(from, t.Position);
                if (d < bestDist)
                {
                    bestDist = d;
                    best = t;
                }
            }

            return best;
        }

        /// <summary>
        /// The nearest cell of open water, for drinking straight from a lake or river
        /// when no stored water is to hand. Returns the walkable cell beside the water
        /// rather than the water itself, since a pawn cannot stand in it.
        /// </summary>
        public bool NearestWaterEdge(Vector2Int from, out Vector2Int cell)
        {
            cell = from;
            float bestDist = float.MaxValue;
            bool found = false;

            for (int y = 0; y < _map.Height; y++)
            {
                for (int x = 0; x < _map.Width; x++)
                {
                    if (_map.Terrain.At(new Vector2Int(x, y)) != TerrainKind.Water)
                        continue;

                    for (int d = 0; d < 4; d++)
                    {
                        Vector2Int side = new Vector2Int(x, y) + Map.CardinalDirections[d];
                        if (!_map.Terrain.Walkable(side))
                            continue;

                        float dist = PathFinder.Heuristic(from, side);
                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            cell = side;
                            found = true;
                        }
                    }
                }
            }

            return found;
        }

        /// <summary>Drop every claim a pawn holds — on death, or when its job is cancelled.</summary>
        public void ReleaseAll(Pawn pawn)
        {
            for (int i = 0; i < _things.Count; i++)
                if (_things[i].ReservedBy == pawn)
                    _things[i].ReservedBy = null;
        }
    }
}
