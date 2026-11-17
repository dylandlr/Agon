using UnityEngine;

namespace Agon.Core
{
    /// <summary>
    /// One colonist. Holds position, needs and health, and ticks them; what the pawn
    /// decides to do lives in the AI layer rather than here, so this stays the state
    /// a saved game is made of.
    /// </summary>
    public class Pawn : ITickable
    {
        public readonly string Id;
        public readonly Map Map;
        public readonly NeedsTracker Needs;

        public Vector2Int Position;

        /// <summary>0..1. Reaching 0 kills the pawn.</summary>
        public float Health = 1f;

        public bool Alive = true;

        public bool Asleep;

        /// <summary>
        /// How far from comfortable the pawn is in degrees: negative is cold,
        /// positive is hot. Set by the temperature system each tick; needs read it.
        /// </summary>
        public float Comfort;

        public TickerType TickerType => TickerType.Normal;

        public Pawn(string id, Map map, Vector2Int position)
        {
            Id = id;
            Map = map;
            Position = position;
            Needs = new NeedsTracker(this);
        }

        public void Tick()
        {
            if (!Alive)
                return;

            Needs.NeedsTrackerTick();
            HealthTick();
        }

        private void HealthTick()
        {
            // A need at zero does damage; several at zero do it faster. Health
            // recovers slowly whenever nothing is critical, so a bad day is
            // survivable but a bad week is not.
            float damage = 0f;

            if (Needs.Food.CurLevel <= 0f) damage += 0.004f;
            if (Needs.Water.CurLevel <= 0f) damage += 0.007f;
            if (Needs.Rest.CurLevel <= 0f) damage += 0.002f;

            if (Mathf.Abs(Comfort) > 12f)
                damage += (Mathf.Abs(Comfort) - 12f) * 0.0004f;

            if (damage > 0f)
            {
                Health = Mathf.Max(0f, Health - damage);
                if (Health <= 0f)
                    Alive = false;
            }
            else if (Health < 1f)
            {
                Health = Mathf.Min(1f, Health + 0.0015f);
            }
        }

        public bool CanReach(Vector2Int cell) => Map.Terrain.Walkable(cell);
    }
}
