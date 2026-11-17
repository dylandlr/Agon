using System.Collections.Generic;
using UnityEngine;

namespace Agon.Core
{
    /// <summary>
    /// Hunger. Falls faster in the cold, because a pawn losing heat burns food to
    /// hold its temperature — the same coupling the HTML build uses, and the reason
    /// a cold snap shows up first as an empty larder rather than as freezing.
    /// </summary>
    public class Need_Food : Need
    {
        public Need_Food(Pawn pawn) : base(pawn) { }

        public override string Label => "Food";

        public override float FallPerTick => 0.006f;

        public override float SeekThreshold => 0.40f;

        protected override float FallRateFactor()
        {
            float warmth = Pawn.Comfort;
            return warmth < 0f ? 1f + Mathf.Min(1f, -warmth * 0.04f) : 1f;
        }
    }

    /// <summary>
    /// Thirst. Falls faster than hunger and has no store to fall back on, which is
    /// why it drives more trips than food does.
    /// </summary>
    public class Need_Water : Need
    {
        public Need_Water(Pawn pawn) : base(pawn) { }

        public override string Label => "Water";

        public override float FallPerTick => 0.009f;

        public override float SeekThreshold => 0.45f;
    }

    /// <summary>
    /// Tiredness. Falls while awake and recovers while asleep, so the pawn's own
    /// state rather than the clock decides the direction.
    /// </summary>
    public class Need_Rest : Need
    {
        public Need_Rest(Pawn pawn) : base(pawn) { }

        public override string Label => "Rest";

        public override float FallPerTick => 0.0075f;

        public override float SeekThreshold => 0.30f;

        /// <summary>How much rest one tick of sleep returns. A full night refills from empty.</summary>
        public const float RecoveryPerTick = 0.055f;

        public override void NeedInterval()
        {
            if (Pawn.Asleep)
                CurLevel += RecoveryPerTick;
            else
                base.NeedInterval();
        }
    }

    /// <summary>A pawn's needs, stepped together once per tick.</summary>
    public class NeedsTracker
    {
        public readonly Need_Food Food;
        public readonly Need_Water Water;
        public readonly Need_Rest Rest;

        private readonly List<Need> _all;

        public NeedsTracker(Pawn pawn)
        {
            Food = new Need_Food(pawn);
            Water = new Need_Water(pawn);
            Rest = new Need_Rest(pawn);

            _all = new List<Need> { Food, Water, Rest };
        }

        public IReadOnlyList<Need> All => _all;

        public void NeedsTrackerTick()
        {
            for (int i = 0; i < _all.Count; i++)
                _all[i].NeedInterval();
        }

        /// <summary>
        /// The need most worth acting on: the lowest one that is below its seek
        /// threshold, or null when nothing is pressing. Comparing by how far each
        /// need has fallen past its own threshold keeps needs with different
        /// thresholds on equal footing.
        /// </summary>
        public Need MostUrgent()
        {
            Need worst = null;
            float worstMargin = 0f;

            for (int i = 0; i < _all.Count; i++)
            {
                Need need = _all[i];
                if (!need.Seeking)
                    continue;

                float margin = need.SeekThreshold - need.CurLevel;
                if (worst == null || margin > worstMargin)
                {
                    worst = need;
                    worstMargin = margin;
                }
            }

            return worst;
        }
    }
}
