using UnityEngine;

namespace Agon.Core
{
    /// <summary>
    /// Carries out one <see cref="Job"/> over however many ticks it takes. A driver is
    /// a two-phase thing in almost every case — walk there, then do the work — so that
    /// shape is built in here and subclasses only say what the work is.
    ///
    /// RimWorld composes drivers from Toils for the same reason; this is the same idea
    /// with the one case that actually recurs spelled out directly.
    /// </summary>
    public abstract class JobDriver
    {
        protected readonly Pawn Pawn;
        protected readonly Job Job;

        private bool _arrived;
        private int _worked;

        protected JobDriver(Pawn pawn, Job job)
        {
            Pawn = pawn;
            Job = job;
        }

        /// <summary>Ticks of work once in place.</summary>
        protected abstract int WorkTicks { get; }

        /// <summary>Called on each tick of work. Returning Done ends the job early.</summary>
        protected virtual JobStatus WorkTick() => JobStatus.Ongoing;

        /// <summary>Called once when the work finishes.</summary>
        protected virtual void Finish() { }

        /// <summary>Whether the job is still worth doing. Checked every tick.</summary>
        protected virtual bool StillValid() =>
            Job.TargetThing == null || !Job.TargetThing.Depleted;

        public JobStatus Tick()
        {
            if (!Pawn.Alive)
                return JobStatus.Failed;

            if (!StillValid())
                return JobStatus.Failed;

            if (!_arrived)
            {
                // adjacent counts as arrived: a bed is stood beside, water is drunk
                // from the bank, and a store is reached over its edge
                if (Pawn.Position == Job.TargetCell ||
                    PathFinder.Heuristic(Pawn.Position, Job.TargetCell) <= PathFinder.DiagonalCost + 0.01f)
                {
                    _arrived = true;
                    Pawn.Path.Stop();
                }
                else
                {
                    if (!Pawn.Path.HasPath && !Pawn.Path.StartPath(Job.TargetCell))
                        return JobStatus.Failed;

                    Pawn.Path.Tick();
                    return JobStatus.Ongoing;
                }
            }

            JobStatus status = WorkTick();
            if (status != JobStatus.Ongoing)
            {
                if (status == JobStatus.Done)
                    Finish();

                return status;
            }

            if (++_worked >= WorkTicks)
            {
                Finish();
                return JobStatus.Done;
            }

            return JobStatus.Ongoing;
        }
    }

    public class JobDriver_Wait : JobDriver
    {
        public JobDriver_Wait(Pawn pawn, Job job) : base(pawn, job) { }

        protected override int WorkTicks => Mathf.Max(1, Job.Duration);
    }

    public class JobDriver_Goto : JobDriver
    {
        public JobDriver_Goto(Pawn pawn, Job job) : base(pawn, job) { }

        protected override int WorkTicks => 1;
    }

    public class JobDriver_Eat : JobDriver
    {
        public const float NutritionPerMeal = 0.65f;

        public JobDriver_Eat(Pawn pawn, Job job) : base(pawn, job) { }

        protected override int WorkTicks => 3;

        protected override void Finish()
        {
            Thing store = Job.TargetThing;
            if (store == null)
                return;

            // never draw more than is there, so the last meal in a store is partial
            // rather than free
            float taken = store.Stock < 0f ? NutritionPerMeal : Mathf.Min(NutritionPerMeal, store.Stock);
            if (store.Stock > 0f)
                store.Stock -= taken;

            Pawn.Needs.Food.Satisfy(taken);
        }
    }

    public class JobDriver_Drink : JobDriver
    {
        public const float HydrationPerDrink = 0.8f;

        public JobDriver_Drink(Pawn pawn, Job job) : base(pawn, job) { }

        protected override int WorkTicks => 2;

        // drinking from open water needs no thing, so a null target is fine here
        protected override bool StillValid() =>
            Job.TargetThing == null || !Job.TargetThing.Depleted;

        protected override void Finish()
        {
            Thing source = Job.TargetThing;

            float taken = HydrationPerDrink;
            if (source != null && source.Stock >= 0f)
            {
                taken = Mathf.Min(taken, source.Stock);
                source.Stock -= taken;
            }

            Pawn.Needs.Water.Satisfy(taken);
        }
    }

    public class JobDriver_Sleep : JobDriver
    {
        public JobDriver_Sleep(Pawn pawn, Job job) : base(pawn, job) { }

        // long enough to refill from empty at the rest need's recovery rate, with the
        // early-out below ending it as soon as the pawn is actually rested
        protected override int WorkTicks => 40;

        protected override JobStatus WorkTick()
        {
            Pawn.Asleep = true;
            return Pawn.Needs.Rest.CurLevel >= 0.98f ? JobStatus.Done : JobStatus.Ongoing;
        }

        protected override void Finish() { Pawn.Asleep = false; }
    }
}
