using System;

namespace Agon.Core
{
    /// <summary>
    /// Which cardinal neighbours a linked tile joins to. The bit values are
    /// RimWorld's (Verse.LinkDirections): the four bits, in this order, form the
    /// index into a 16-cell linked atlas.
    /// </summary>
    [Flags]
    public enum LinkDirections : byte
    {
        None  = 0,
        Up    = 1,
        Right = 2,
        Down  = 4,
        Left  = 8,

        All   = Up | Right | Down | Left
    }

    /// <summary>
    /// What a tile links with. A tile joins a neighbour when the neighbour's flags
    /// and the tile's own link flags intersect — RimWorld's Graphic_Linked.ShouldLinkWith.
    /// </summary>
    [Flags]
    public enum LinkFlags
    {
        None    = 0,
        MapEdge = 1,
        Rock    = 2,
        Wall    = 4,
        Door    = 8,

        /// <summary>What a built wall links with: other walls, natural rock, doors and the map edge.</summary>
        WallLike = MapEdge | Rock | Wall | Door
    }
}
