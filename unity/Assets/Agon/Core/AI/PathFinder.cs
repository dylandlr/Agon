using System;
using System.Collections.Generic;
using UnityEngine;

namespace Agon.Core
{
    /// <summary>
    /// A* over the cell grid, eight-directional.
    ///
    /// Two rules make grid paths read as sensible movement rather than as geometry:
    ///
    ///   1. **No corner cutting.** A diagonal step is allowed only when both of the
    ///      cells it passes between are walkable, so a pawn never slips through the
    ///      join between two wall corners. RimWorld does the same, and without it
    ///      pawns walk through sealed rooms along their diagonals.
    ///   2. **Octile distance as the heuristic.** The straight-line distance
    ///      underestimates grid movement badly enough to make A* explore most of the
    ///      map; octile is the true cost of an unobstructed eight-way walk, so it
    ///      stays admissible while actually pruning.
    ///
    /// The open set is a binary heap keyed on f = g + h. Closed cells are tracked in
    /// a generation-stamped array rather than a HashSet, so repeated searches on the
    /// same map allocate nothing after the first.
    /// </summary>
    public class PathFinder
    {
        /// <summary>Cost of a cardinal step, before terrain. Diagonals cost √2 times this.</summary>
        public const float StepCost = 1f;

        public const float DiagonalCost = 1.41421356f;

        /// <summary>Cells a single search may expand before giving up.</summary>
        public int MaxExpansions = 20000;

        private readonly Map _map;

        private readonly float[] _g;
        private readonly int[] _parent;
        private readonly int[] _stamp;
        private readonly bool[] _closed;
        private int _generation;

        private readonly int[] _heap;
        private readonly float[] _heapKey;
        private int _heapCount;

        public PathFinder(Map map)
        {
            _map = map;
            int n = map.CellCount;

            _g = new float[n];
            _parent = new int[n];
            _stamp = new int[n];
            _closed = new bool[n];
            _heap = new int[n + 1];
            _heapKey = new float[n + 1];
        }

        /// <summary>
        /// Find a walkable path from <paramref name="start"/> to <paramref name="dest"/>.
        /// On success <paramref name="result"/> holds the cells to walk, starting with
        /// the first step and ending with the destination; the start cell is not
        /// included, since the pawn is already standing on it.
        /// </summary>
        public bool FindPath(Vector2Int start, Vector2Int dest, List<Vector2Int> result)
        {
            result.Clear();

            if (!_map.InBounds(start) || !_map.InBounds(dest))
                return false;

            if (start == dest)
                return true;

            if (!_map.Terrain.Walkable(dest))
                return false;

            _generation++;
            _heapCount = 0;

            int startIndex = _map.CellToIndex(start);
            int destIndex = _map.CellToIndex(dest);

            Touch(startIndex);
            _g[startIndex] = 0f;
            _parent[startIndex] = -1;
            HeapPush(startIndex, Heuristic(start, dest));

            int expansions = 0;

            while (_heapCount > 0)
            {
                int current = HeapPop();

                if (_closed[current])
                    continue;

                _closed[current] = true;

                if (current == destIndex)
                {
                    Reconstruct(destIndex, result);
                    return true;
                }

                if (++expansions > MaxExpansions)
                    return false;

                Vector2Int cell = _map.IndexToCell(current);

                for (int dir = 0; dir < 8; dir++)
                {
                    Vector2Int step = Neighbours[dir];
                    Vector2Int next = cell + step;

                    if (!_map.InBounds(next) || !_map.Terrain.Walkable(next))
                        continue;

                    bool diagonal = step.x != 0 && step.y != 0;

                    // no corner cutting: both flanking cells must be open
                    if (diagonal &&
                        (!_map.Terrain.Walkable(new Vector2Int(cell.x + step.x, cell.y)) ||
                         !_map.Terrain.Walkable(new Vector2Int(cell.x, cell.y + step.y))))
                        continue;

                    int nextIndex = _map.CellToIndex(next);
                    Touch(nextIndex);

                    if (_closed[nextIndex])
                        continue;

                    float cost = (diagonal ? DiagonalCost : StepCost) * _map.Terrain.MoveCost(next);
                    float tentative = _g[current] + cost;

                    if (tentative < _g[nextIndex])
                    {
                        _g[nextIndex] = tentative;
                        _parent[nextIndex] = current;
                        HeapPush(nextIndex, tentative + Heuristic(next, dest));
                    }
                }
            }

            return false;
        }

        /// <summary>
        /// Octile distance: the exact cost of an eight-way walk over open ground, which
        /// keeps the heuristic admissible while still pruning hard.
        /// </summary>
        public static float Heuristic(Vector2Int a, Vector2Int b)
        {
            int dx = Mathf.Abs(a.x - b.x);
            int dy = Mathf.Abs(a.y - b.y);
            int min = Mathf.Min(dx, dy);

            return (dx + dy - 2 * min) * StepCost + min * DiagonalCost;
        }

        private static readonly Vector2Int[] Neighbours =
        {
            new Vector2Int(0, 1), new Vector2Int(1, 0), new Vector2Int(0, -1), new Vector2Int(-1, 0),
            new Vector2Int(1, 1), new Vector2Int(1, -1), new Vector2Int(-1, -1), new Vector2Int(-1, 1)
        };

        /// <summary>Lazily reset a cell's bookkeeping for this search generation.</summary>
        private void Touch(int index)
        {
            if (_stamp[index] == _generation)
                return;

            _stamp[index] = _generation;
            _g[index] = float.MaxValue;
            _parent[index] = -1;
            _closed[index] = false;
        }

        private void Reconstruct(int destIndex, List<Vector2Int> result)
        {
            int at = destIndex;
            while (at != -1 && _parent[at] != -1)
            {
                result.Add(_map.IndexToCell(at));
                at = _parent[at];
            }

            result.Reverse();
        }

        private void HeapPush(int index, float key)
        {
            int i = ++_heapCount;
            _heap[i] = index;
            _heapKey[i] = key;

            while (i > 1)
            {
                int parent = i >> 1;
                if (_heapKey[parent] <= _heapKey[i])
                    break;

                Swap(parent, i);
                i = parent;
            }
        }

        private int HeapPop()
        {
            int top = _heap[1];

            _heap[1] = _heap[_heapCount];
            _heapKey[1] = _heapKey[_heapCount];
            _heapCount--;

            int i = 1;
            while (true)
            {
                int left = i << 1, right = left + 1, best = i;

                if (left <= _heapCount && _heapKey[left] < _heapKey[best]) best = left;
                if (right <= _heapCount && _heapKey[right] < _heapKey[best]) best = right;

                if (best == i)
                    break;

                Swap(best, i);
                i = best;
            }

            return top;
        }

        private void Swap(int a, int b)
        {
            int ti = _heap[a]; _heap[a] = _heap[b]; _heap[b] = ti;
            float tk = _heapKey[a]; _heapKey[a] = _heapKey[b]; _heapKey[b] = tk;
        }
    }
}
