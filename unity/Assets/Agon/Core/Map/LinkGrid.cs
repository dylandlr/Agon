using UnityEngine;

namespace Agon.Core
{
    /// <summary>
    /// What each cell offers neighbouring tiles to link with — RimWorld's Verse.LinkGrid.
    /// Derived from the edifice grid, and kept in step by <see cref="Recompute"/>.
    /// </summary>
    public class LinkGrid
    {
        private readonly Map _map;
        private readonly LinkFlags[] _flags;

        public LinkGrid(Map map)
        {
            _map = map;
            _flags = new LinkFlags[map.CellCount];
        }

        /// <summary>
        /// The link flags at a cell. Off-map reads as <see cref="LinkFlags.MapEdge"/>:
        /// RimWorld does the same, so a wall run meeting the map edge closes against it
        /// instead of ending in a stub.
        /// </summary>
        public LinkFlags FlagsAt(Vector2Int c)
        {
            if (!_map.InBounds(c))
                return LinkFlags.MapEdge;

            return _flags[_map.CellToIndex(c)];
        }

        public void Recompute(Vector2Int c)
        {
            if (!_map.InBounds(c))
                return;

            _flags[_map.CellToIndex(c)] = FlagsFor(_map.Edifices.KindAt(c));
        }

        public void RecomputeAll()
        {
            for (int i = 0; i < _flags.Length; i++)
                _flags[i] = FlagsFor(_map.Edifices.KindAt(_map.IndexToCell(i)));
        }

        private static LinkFlags FlagsFor(EdificeKind kind)
        {
            switch (kind)
            {
                case EdificeKind.Wall: return LinkFlags.Wall;
                case EdificeKind.Door: return LinkFlags.Door;
                case EdificeKind.NaturalRock: return LinkFlags.Rock;
                default: return LinkFlags.None;
            }
        }

        /// <summary>
        /// Whether a tile with <paramref name="linkFlags"/> joins the cell at
        /// <paramref name="c"/> — RimWorld's Graphic_Linked.ShouldLinkWith, which
        /// intersects the neighbour's flags with the tile's own.
        /// </summary>
        public bool ShouldLinkWith(Vector2Int c, LinkFlags linkFlags) =>
            (FlagsAt(c) & linkFlags) != LinkFlags.None;

        /// <summary>
        /// The link mask for a cell: the four cardinal neighbours tested in
        /// LinkDirections bit order (Up=1, Right=2, Down=4, Left=8). This is the
        /// index into a 16-cell linked atlas, exactly as in
        /// Graphic_Linked.LinkedDrawMatFrom.
        /// </summary>
        public LinkDirections LinksAt(Vector2Int c, LinkFlags linkFlags)
        {
            int mask = 0;
            int bit = 1;

            for (int i = 0; i < 4; i++)
            {
                if (ShouldLinkWith(c + Map.CardinalDirections[i], linkFlags))
                    mask += bit;

                bit *= 2;
            }

            return (LinkDirections)mask;
        }
    }
}
