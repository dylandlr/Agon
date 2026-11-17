using System;
using UnityEngine;

namespace Agon.Core
{
    /// <summary>
    /// One playable map: a fixed-size grid of cells plus the grids laid over it.
    /// Cells are addressed as <see cref="Vector2Int"/> with (0,0) at the lower-left,
    /// and flattened row-major (index = y * Width + x) by <see cref="CellToIndex"/>.
    /// </summary>
    public class Map
    {
        public readonly int Width;
        public readonly int Height;

        public readonly TerrainGrid Terrain;
        public readonly EdificeGrid Edifices;
        public readonly LinkGrid Links;
        public readonly ThingGrid Things;
        public readonly PathFinder Pathing;

        public int CellCount => Width * Height;

        public Map(int width, int height)
        {
            if (width <= 0 || height <= 0)
                throw new ArgumentOutOfRangeException(nameof(width), "Map size must be positive.");

            Width = width;
            Height = height;

            Terrain = new TerrainGrid(this);
            Edifices = new EdificeGrid(this);
            Links = new LinkGrid(this);
            Things = new ThingGrid(this);
            Pathing = new PathFinder(this);
        }

        public bool InBounds(Vector2Int c) =>
            c.x >= 0 && c.y >= 0 && c.x < Width && c.y < Height;

        public bool InBounds(int x, int y) =>
            x >= 0 && y >= 0 && x < Width && y < Height;

        public int CellToIndex(Vector2Int c) => c.y * Width + c.x;

        public int CellToIndex(int x, int y) => y * Width + x;

        public Vector2Int IndexToCell(int index) =>
            new Vector2Int(index % Width, index / Width);

        /// <summary>The four cardinal neighbours, in LinkDirections bit order: Up, Right, Down, Left.</summary>
        public static readonly Vector2Int[] CardinalDirections =
        {
            new Vector2Int(0, 1),
            new Vector2Int(1, 0),
            new Vector2Int(0, -1),
            new Vector2Int(-1, 0)
        };

        /// <summary>
        /// The four diagonals, in the order RimWorld's Graphic_LinkedCornerFiller walks them
        /// (GenAdj.DiagonalDirectionsAround): SW, NW, NE, SE. The corner filler's guard pairs
        /// follow this order, so keep the two in step.
        /// </summary>
        public static readonly Vector2Int[] DiagonalDirections =
        {
            new Vector2Int(-1, -1),
            new Vector2Int(-1, 1),
            new Vector2Int(1, 1),
            new Vector2Int(1, -1)
        };
    }
}
