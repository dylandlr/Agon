using UnityEngine;
using Agon.Core;

namespace Agon.Rendering
{
    /// <summary>
    /// Draws a 16-cell linked-wall atlas at runtime, so walls render with no
    /// authored art. Replacing this with a painted sheet needs nothing but a texture
    /// laid out the same way: 4x4 cells, cell index = Up(1) | Right(2) | Down(4) | Left(8),
    /// row = index / 4 counted from the bottom.
    ///
    /// Each cell is drawn as the shape the HTML build uses: a rounded core square
    /// that grows an arm toward every linked neighbour, so adjacent tiles' arms meet
    /// and the run reads as one mass. A lighter strip along the south edge is the
    /// wall's face catching the light, which is what gives the top-down view its
    /// sense of height.
    ///
    /// Only the middle 75% of each cell is ever sampled (see
    /// <see cref="LinkedAtlas.TexPadding"/>), so the art is drawn into that region
    /// and the arms run out through the border, which is culled. That border is why
    /// neighbouring cells never bleed into each other.
    /// </summary>
    public static class WallAtlasGenerator
    {
        /// <summary>Pixels per atlas cell. The sampled tile is 75% of this.</summary>
        public const int CellPixels = 64;

        public const int AtlasPixels = CellPixels * LinkedAtlas.Columns;

        public static Texture2D Generate(WallMaterial material)
        {
            WallMaterialDef def = WallMaterialDef.Of(material);

            Color body = def.Color;
            Color face = Lighten(body, 0.30f);
            Color ink = new Color32(0x1C, 0x11, 0x0B, 0xFF);

            var tex = new Texture2D(AtlasPixels, AtlasPixels, TextureFormat.RGBA32, false)
            {
                name = "AgonWallAtlas_" + material,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };

            var pixels = new Color32[AtlasPixels * AtlasPixels];
            for (int i = 0; i < pixels.Length; i++)
                pixels[i] = new Color32(0, 0, 0, 0);

            for (int index = 0; index < LinkedAtlas.CellCount; index++)
                DrawCell(pixels, index, (LinkDirections)index, body, face, ink);

            tex.SetPixels32(pixels);
            tex.Apply(false, false);
            return tex;
        }

        private static void DrawCell(Color32[] pixels, int index, LinkDirections links,
                                     Color body, Color face, Color ink)
        {
            int cellX = (index % LinkedAtlas.Columns) * CellPixels;
            int cellY = (index / LinkedAtlas.Columns) * CellPixels;

            // The sampled region inside this cell, in pixels.
            int pad = Mathf.RoundToInt(LinkedAtlas.TexPadding * CellPixels / LinkedAtlas.CellSize);
            int tile = CellPixels - 2 * pad;

            // Core of the wall block: the middle ~58% of the sampled tile, matching
            // the HTML build's 28-of-48 core.
            int coreInset = Mathf.RoundToInt(tile * 0.208f);
            int outline = Mathf.Max(1, Mathf.RoundToInt(tile * 0.0625f));

            int x0 = pad + coreInset, x1 = pad + tile - coreInset;
            int y0 = pad + coreInset, y1 = pad + tile - coreInset;

            // Arms reach past the sampled edge into the culled border, so a tile's arm
            // meets its neighbour's with no seam.
            int armLeft   = (links & LinkDirections.Left)  != 0 ? 0 : x0;
            int armRight  = (links & LinkDirections.Right) != 0 ? CellPixels : x1;
            int armDown   = (links & LinkDirections.Down)  != 0 ? 0 : y0;
            int armUp     = (links & LinkDirections.Up)    != 0 ? CellPixels : y1;

            // Outline first, then the body inset into it.
            FillRect(pixels, cellX, cellY, x0 - outline, y0 - outline, x1 + outline, y1 + outline, ink);
            FillRect(pixels, cellX, cellY, armLeft, y0 - outline, armRight, y1 + outline, ink);
            FillRect(pixels, cellX, cellY, x0 - outline, armDown, x1 + outline, armUp, ink);

            FillRect(pixels, cellX, cellY, x0, y0, x1, y1, body);
            FillRect(pixels, cellX, cellY, armLeft, y0, armRight, y1, body);
            FillRect(pixels, cellX, cellY, x0, armDown, x1, armUp, body);

            // The south face: a lit strip along every south-facing edge. The core shows
            // one unless the wall continues south; an east or west arm carries its own.
            int faceHeight = Mathf.RoundToInt(tile * 0.21f);

            if ((links & LinkDirections.Down) == 0)
                FillRect(pixels, cellX, cellY, x0, y0, x1, y0 + faceHeight, face);

            if ((links & LinkDirections.Right) != 0)
                FillRect(pixels, cellX, cellY, x1, y0, armRight, y0 + faceHeight, face);

            if ((links & LinkDirections.Left) != 0)
                FillRect(pixels, cellX, cellY, armLeft, y0, x0, y0 + faceHeight, face);
        }

        private static void FillRect(Color32[] pixels, int cellX, int cellY,
                                     int x0, int y0, int x1, int y1, Color color)
        {
            x0 = Mathf.Max(0, x0);
            y0 = Mathf.Max(0, y0);
            x1 = Mathf.Min(CellPixels, x1);
            y1 = Mathf.Min(CellPixels, y1);

            var c32 = (Color32)color;

            for (int y = y0; y < y1; y++)
            {
                int row = (cellY + y) * AtlasPixels + cellX;
                for (int x = x0; x < x1; x++)
                    pixels[row + x] = c32;
            }
        }

        private static Color Lighten(Color c, float t) =>
            new Color(
                Mathf.Lerp(c.r, 1f, t),
                Mathf.Lerp(c.g, 1f, t),
                Mathf.Lerp(c.b, 1f, t),
                c.a);
    }
}
