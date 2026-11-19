using UnityEngine;

namespace Agon.Core
{
    /// <summary>What occupies a cell structurally. One edifice per cell, as in RimWorld.</summary>
    public enum EdificeKind : byte
    {
        None = 0,
        Wall = 1,
        Door = 2,
        NaturalRock = 3
    }

    /// <summary>
    /// The structures standing on the map: walls, doors and natural rock, one per cell,
    /// each with the material it is built from.
    /// </summary>
    public class EdificeGrid
    {
        private readonly Map _map;
        private readonly EdificeKind[] _kinds;
        private readonly WallMaterial[] _materials;

        /// <summary>Raised when a cell changes, so renderers can rebuild. Carries the changed cell.</summary>
        public event System.Action<Vector2Int> Changed;

        public EdificeGrid(Map map)
        {
            _map = map;
            _kinds = new EdificeKind[map.CellCount];
            _materials = new WallMaterial[map.CellCount];
        }

        public EdificeKind KindAt(Vector2Int c) =>
            _map.InBounds(c) ? _kinds[_map.CellToIndex(c)] : EdificeKind.None;

        public WallMaterial MaterialAt(Vector2Int c) =>
            _map.InBounds(c) ? _materials[_map.CellToIndex(c)] : WallMaterial.Wattle;

        public bool IsWall(Vector2Int c) => KindAt(c) == EdificeKind.Wall;

        public bool IsDoor(Vector2Int c) => KindAt(c) == EdificeKind.Door;

        /// <summary>Anything that reads as solid structure — wall, door or natural rock.</summary>
        public bool IsStructure(Vector2Int c) => KindAt(c) != EdificeKind.None;

        public void Set(Vector2Int c, EdificeKind kind, WallMaterial material = WallMaterial.Wattle)
        {
            if (!_map.InBounds(c))
                return;

            int i = _map.CellToIndex(c);
            if (_kinds[i] == kind && _materials[i] == material)
                return;

            _kinds[i] = kind;
            _materials[i] = material;
            _map.Links.Recompute(c);
            Changed?.Invoke(c);
        }

        public void Clear(Vector2Int c) => Set(c, EdificeKind.None);
    }
}
