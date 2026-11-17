using System;
using UnityEngine;

namespace Agon.Core
{
    /// <summary>
    /// One drive a pawn has to keep satisfied — food, water, rest. Modelled on
    /// RimWorld's RimWorld.Need: a level in 0..1 that falls on its own and is raised
    /// by acting on it, with thresholds the AI reads to decide when to care.
    ///
    /// Timing differs from RimWorld on purpose. RimWorld's day is 60000 ticks and it
    /// runs NeedInterval every 150; Agon's day is <see cref="TickManager.TicksPerDay"/>
    /// = 50 ticks, so needs are stepped every tick and the fall rates below are per
    /// tick. Rates match the HTML build so the two implementations behave alike.
    /// </summary>
    public abstract class Need
    {
        protected readonly Pawn Pawn;

        private float _curLevel;

        protected Need(Pawn pawn)
        {
            Pawn = pawn;
            _curLevel = InitialLevel;
        }

        public abstract string Label { get; }

        /// <summary>How much this need falls per tick when nothing is done about it.</summary>
        public abstract float FallPerTick { get; }

        /// <summary>Below this the pawn starts wanting to act on the need.</summary>
        public virtual float SeekThreshold => 0.35f;

        /// <summary>Below this the need is doing the pawn harm.</summary>
        public virtual float CriticalThreshold => 0.05f;

        public virtual float InitialLevel => 0.8f;

        public float MaxLevel => 1f;

        public float CurLevel
        {
            get => _curLevel;
            set => _curLevel = Mathf.Clamp(value, 0f, MaxLevel);
        }

        public float CurLevelPercentage
        {
            get => CurLevel / MaxLevel;
            set => CurLevel = value * MaxLevel;
        }

        public bool Seeking => CurLevel < SeekThreshold;

        public bool Critical => CurLevel < CriticalThreshold;

        /// <summary>Advance the need by one tick. Override to add behaviour beyond the fall.</summary>
        public virtual void NeedInterval()
        {
            CurLevel -= FallPerTick * FallRateFactor();
        }

        /// <summary>
        /// Multiplier on the fall rate for this tick. The base is 1; subclasses scale
        /// it by circumstance, such as cold making a pawn burn food faster.
        /// </summary>
        protected virtual float FallRateFactor() => 1f;

        public void Satisfy(float amount) => CurLevel += amount;
    }
}
