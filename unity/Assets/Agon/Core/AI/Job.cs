using UnityEngine;

namespace Agon.Core
{
    public enum JobDef : byte
    {
        Wait = 0,
        Goto = 1,
        Eat = 2,
        Drink = 3,
        Sleep = 4
    }

    /// <summary>
    /// One piece of work a pawn has taken on: what to do and what to do it to.
    /// Deliberately a plain record — the behaviour lives in <see cref="JobDriver"/>,
    /// so a job can be queued, cancelled or saved without carrying any state machine
    /// with it. This is RimWorld's Job / JobDriver split.
    /// </summary>
    public class Job
    {
        public readonly JobDef Def;

        /// <summary>Where the work happens. For a Thing job this is the thing's cell.</summary>
        public Vector2Int TargetCell;

        public Thing TargetThing;

        /// <summary>Ticks to spend once in place. 0 means the driver decides.</summary>
        public int Duration;

        public Job(JobDef def, Vector2Int targetCell, Thing targetThing = null, int duration = 0)
        {
            Def = def;
            TargetCell = targetCell;
            TargetThing = targetThing;
            Duration = duration;
        }

        public override string ToString() =>
            Def + " @" + TargetCell + (TargetThing != null ? " (" + TargetThing.Kind + ")" : "");
    }

    public enum JobStatus : byte
    {
        /// <summary>Still working; tick it again next tick.</summary>
        Ongoing = 0,
        Done = 1,
        /// <summary>Could not be carried out — unreachable, gone, or taken by someone else.</summary>
        Failed = 2
    }
}
