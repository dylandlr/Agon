using System.Collections.Generic;
using UnityEngine;

namespace Agon.Core
{
    /// <summary>
    /// Walks a pawn along a path, one cell at a time, paying the terrain's cost for
    /// each step. Movement is accumulated rather than instant so that rough ground
    /// genuinely slows a pawn down, and so a pawn's position is always a whole cell —
    /// which is what the pathfinder, the job drivers and reservation all assume.
    ///
    /// The path is re-checked as it is walked: anything that blocks a later cell after
    /// the path was found, such as a wall going up, makes the follower recompute from
    /// where it stands instead of walking into it.
    /// </summary>
    public class PathFollower
    {
        /// <summary>Movement earned per tick. One over this is the ticks per open-ground cell.</summary>
        public const float SpeedPerTick = 0.34f;

        private readonly Pawn _pawn;
        private readonly List<Vector2Int> _path = new List<Vector2Int>();
        private int _index;
        private float _progress;

        public PathFollower(Pawn pawn) { _pawn = pawn; }

        public bool HasPath => _index < _path.Count;

        public Vector2Int Destination => _path.Count > 0 ? _path[_path.Count - 1] : _pawn.Position;

        /// <summary>Cells still to walk, for display and for tests.</summary>
        public int Remaining => Mathf.Max(0, _path.Count - _index);

        public bool StartPath(Vector2Int dest)
        {
            Stop();

            if (dest == _pawn.Position)
                return true;

            if (!_pawn.Map.Pathing.FindPath(_pawn.Position, dest, _path))
            {
                _path.Clear();
                return false;
            }

            _index = 0;
            _progress = 0f;
            return _path.Count > 0;
        }

        public void Stop()
        {
            _path.Clear();
            _index = 0;
            _progress = 0f;
        }

        public void Tick()
        {
            if (!HasPath)
                return;

            Vector2Int next = _path[_index];

            // something may have blocked the way since the path was found
            if (!_pawn.Map.Terrain.Walkable(next))
            {
                Vector2Int dest = Destination;
                if (!StartPath(dest))
                    Stop();

                return;
            }

            bool diagonal = next.x != _pawn.Position.x && next.y != _pawn.Position.y;
            float cost = (diagonal ? PathFinder.DiagonalCost : PathFinder.StepCost)
                       * _pawn.Map.Terrain.MoveCost(next);

            _progress += SpeedPerTick;

            if (_progress < cost)
                return;

            _progress -= cost;
            _pawn.Position = next;
            _index++;

            if (!HasPath)
                Stop();
        }
    }
}
