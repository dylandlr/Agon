using UnityEngine;
using Agon.Core;

namespace Agon.Rendering
{
    /// <summary>
    /// UV lookup into a linked-wall atlas: one texture holding 16 sub-tiles in a
    /// 4x4 grid, one per combination of linked cardinal neighbours.
    ///
    /// Ported from RimWorld's Verse.MaterialAtlasPool.MaterialAtlas, which builds
    /// 16 materials from one texture with
    ///     scale  = (0.1875, 0.1875)
    ///     offset = (i % 4 * 0.25 + 0.03125, i / 4 * 0.25 + 0.03125)
    /// i.e. each cell is a quarter of the sheet, inset by 1/32 on every side, so
    /// only the middle 75% of the cell is sampled. The inset is the point: the
    /// authored art may run right to the cell edge, and the culled border keeps
    /// the neighbouring cell's art from bleeding in when the GPU filters.
    ///
    /// Cell layout, with (0,0) at the bottom-left in UV space:
    ///     index = Up(1) | Right(2) | Down(4) | Left(8)
    ///     column = index % 4, row = index / 4
    /// Authoring a sheet therefore means laying the 16 pieces out in that order.
    /// </summary>
    public static class LinkedAtlas
    {
        public const int CellCount = 16;
        public const int Columns = 4;

        /// <summary>Width/height of one cell in UV space, before the padding inset.</summary>
        public const float CellSize = 0.25f;

        /// <summary>RimWorld's MaterialAtlas.TexPadding — the inset culled from each cell edge.</summary>
        public const float TexPadding = 0.03125f;

        /// <summary>Sampled extent of one cell: 0.25 - 2 * 0.03125 = 0.1875.</summary>
        public const float SampledSize = CellSize - 2f * TexPadding;

        /// <summary>
        /// The UV rect sampled for a given link combination. x/y are the lower-left
        /// corner, width/height are <see cref="SampledSize"/>.
        /// </summary>
        public static Rect UvRect(LinkDirections links)
        {
            int index = (int)links & (CellCount - 1);
            float x = (index % Columns) * CellSize + TexPadding;
            float y = (index / Columns) * CellSize + TexPadding;
            return new Rect(x, y, SampledSize, SampledSize);
        }

        /// <summary>
        /// A single point well inside a cell, used to fill a corner with flat colour.
        /// RimWorld's Graphic_LinkedCornerFiller samples (0.5, 0.6) of the whole sheet
        /// for all four vertices of its cover quad; the equivalent here is a point
        /// inside the fully-enclosed cell (all four neighbours linked), which is solid
        /// wall by construction, so the patch always matches the wall's body colour.
        /// </summary>
        public static Vector2 SolidUv()
        {
            Rect r = UvRect(LinkDirections.All);
            return new Vector2(r.x + r.width * 0.5f, r.y + r.height * 0.5f);
        }
    }
}
