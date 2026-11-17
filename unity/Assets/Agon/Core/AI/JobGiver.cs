using UnityEngine;

namespace Agon.Core
{
    /// <summary>
    /// Decides what a pawn should do next. Needs come first and in order of how far
    /// each has fallen past its own threshold, so a pawn dying of thirst does not stop
    /// to eat. When nothing is pressing the pawn waits, which is a job like any other
    /// rather than a special idle state — that way there is always exactly one thing
    /// being ticked.
    ///
    /// A job is only handed out if its target can actually be reached and claimed;
    /// an unreachable store is no reason to walk halfway across the map.
    /// </summary>
    public static class JobGiver
    {
        public static Job GiveJob(Pawn pawn)
        {
            Need urgent = pawn.Needs.MostUrgent();

            if (urgent != null)
            {
                Job job = JobForNeed(pawn, urgent);
                if (job != null)
                    return job;

                // the most urgent need cannot be served here: try the others before
                // giving up, so a colony with no water still eats
                foreach (Need need in pawn.Needs.All)
                {
                    if (need == urgent || !need.Seeking)
                        continue;

                    job = JobForNeed(pawn, need);
                    if (job != null)
                        return job;
                }
            }

            return new Job(JobDef.Wait, pawn.Position, null, 10);
        }

        private static Job JobForNeed(Pawn pawn, Need need)
        {
            ThingGrid things = pawn.Map.Things;

            if (need == pawn.Needs.Food)
            {
                Thing store = things.NearestUsable(ThingKind.FoodStore, pawn.Position, pawn);
                if (store == null)
                    return null;

                return Claim(pawn, new Job(JobDef.Eat, store.Position, store));
            }

            if (need == pawn.Needs.Water)
            {
                Thing source = things.NearestUsable(ThingKind.WaterSource, pawn.Position, pawn);
                if (source != null)
                    return Claim(pawn, new Job(JobDef.Drink, source.Position, source));

                // nothing stored: drink from the nearest open water instead
                if (things.NearestWaterEdge(pawn.Position, out Vector2Int edge))
                    return new Job(JobDef.Drink, edge);

                return null;
            }

            if (need == pawn.Needs.Rest)
            {
                Thing bed = things.NearestUsable(ThingKind.Bed, pawn.Position, pawn);
                if (bed != null)
                    return Claim(pawn, new Job(JobDef.Sleep, bed.Position, bed));

                // no bed free: sleep where you stand, which is worse but not nothing
                return new Job(JobDef.Sleep, pawn.Position);
            }

            return null;
        }

        private static Job Claim(Pawn pawn, Job job)
        {
            if (job.TargetThing != null)
                job.TargetThing.ReservedBy = pawn;

            return job;
        }

        public static JobDriver DriverFor(Pawn pawn, Job job)
        {
            switch (job.Def)
            {
                case JobDef.Eat: return new JobDriver_Eat(pawn, job);
                case JobDef.Drink: return new JobDriver_Drink(pawn, job);
                case JobDef.Sleep: return new JobDriver_Sleep(pawn, job);
                case JobDef.Goto: return new JobDriver_Goto(pawn, job);
                default: return new JobDriver_Wait(pawn, job);
            }
        }
    }

    /// <summary>
    /// Holds the one job a pawn is doing and asks for another when it ends. Separating
    /// this from the pawn keeps "what am I doing" in one place, which is what makes a
    /// stuck pawn diagnosable: <see cref="CurrentJob"/> and <see cref="LastFailed"/>
    /// say where it is and what it last could not do.
    /// </summary>
    public class JobTracker
    {
        private readonly Pawn _pawn;
        private JobDriver _driver;

        public Job CurrentJob { get; private set; }

        /// <summary>The last job that failed, kept for display and debugging.</summary>
        public Job LastFailed { get; private set; }

        public int JobsDone { get; private set; }

        public JobTracker(Pawn pawn) { _pawn = pawn; }

        public void Tick()
        {
            if (CurrentJob == null)
                StartNewJob();

            if (_driver == null)
                return;

            JobStatus status = _driver.Tick();

            if (status == JobStatus.Ongoing)
                return;

            if (status == JobStatus.Done)
                JobsDone++;
            else
                LastFailed = CurrentJob;

            EndCurrentJob();

            // take the next job in the same tick, so a finished job never costs an
            // idle tick — a pawn that keeps failing would otherwise crawl
            StartNewJob();
        }

        public void EndCurrentJob()
        {
            _pawn.Map.Things.ReleaseAll(_pawn);
            _pawn.Path.Stop();
            _pawn.Asleep = false;
            CurrentJob = null;
            _driver = null;
        }

        private void StartNewJob()
        {
            CurrentJob = JobGiver.GiveJob(_pawn);
            _driver = CurrentJob != null ? JobGiver.DriverFor(_pawn, CurrentJob) : null;
        }
    }
}
