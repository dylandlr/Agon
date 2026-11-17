using UnityEngine;

namespace Agon.Core
{
    public enum TerrainKind : byte
    {
        Soil = 0,
        Grass = 1,
        Forest = 2,
        Rock = 3,
        Water = 4,
        Sand = 5,
        Snow = 6,
        Floor = 7
    }

    /// <summary>The ground under each cell. Walkability and movement cost live here.</summary>
    public class TerrainGrid
    {
        private readonly Map _map;
        private readonly TerrainKind[] _kinds;

        public TerrainGrid(Map map)
        {
            _map = map;
            _kinds = new TerrainKind[map.CellCount];
        }

        public TerrainKind At(Vector2Int c) =>
            _map.InBounds(c) ? _kinds[_map.CellToIndex(c)] : TerrainKind.Rock;

        public void Set(Vector2Int c, TerrainKind kind)
        {
            if (_map.InBounds(c))
                _kinds[_map.CellToIndex(c)] = kind;
        }

        public void Fill(TerrainKind kind)
        {
            for (int i = 0; i < _kinds.Length; i++)
                _kinds[i] = kind;
        }

        /// <summary>Whether a pawn can stand here: passable terrain with no structure on it.</summary>
        public bool Walkable(Vector2Int c)
        {
            if (!_map.InBounds(c))
                return false;

            if (_map.Edifices.IsStructure(c) && !_map.Edifices.IsDoor(c))
                return false;

            TerrainKind t = At(c);
            return t != TerrainKind.Rock && t != TerrainKind.Water;
        }

        /// <summary>Extra ticks a pawn spends crossing this cell, beyond the base cost.</summary>
        public float MoveCost(Vector2Int c)
        {
            switch (At(c))
            {
                case TerrainKind.Forest: return 1.6f;
                case TerrainKind.Sand: return 1.3f;
                case TerrainKind.Snow: return 1.5f;
                case TerrainKind.Floor: return 0.8f;
                default: return 1f;
            }
        }
    }
}
