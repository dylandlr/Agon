using System.Collections.Generic;
using UnityEngine;
using Agon.Core;

namespace Agon.Rendering
{
    /// <summary>
    /// Builds the wall mesh for a map: one quad per wall cell, its UVs picked from a
    /// 16-cell linked atlas by which cardinal neighbours it joins, plus the corner
    /// patches that close the notch where four linked tiles meet.
    ///
    /// This is a port of RimWorld's Verse.Graphic_Linked (the atlas lookup) and
    /// Verse.Graphic_LinkedCornerFiller (the patches), which is what walls actually
    /// draw with. The two rules together are what make a run of wall read as one
    /// connected mass rather than a row of separate tiles:
    ///
    ///   1. Linking. Four cardinal neighbours are tested in LinkDirections bit order
    ///      (Up=1, Right=2, Down=4, Left=8) and the resulting 0..15 mask indexes the
    ///      atlas. See <see cref="LinkGrid.LinksAt"/>.
    ///
    ///   2. Corner filling. For each diagonal neighbour that links, when *both*
    ///      flanking cardinals link too, the tile draws an extra half-size quad over
    ///      its own quadrant facing that diagonal, sampling a solid point of the
    ///      texture. All four tiles around a shared corner do this, so their four
    ///      quadrants together cover the gap the inset tile art leaves at the corner.
    ///      Without it, every 2x2 block of wall shows a hole at its centre.
    ///
    /// Geometry note: RimWorld offsets the cover quad by
    ///     DistCenterCorner - CoverSizeCornerCorner * 0.5 = |(.5,.5)| - |(.5,.5)| * .5
    /// which is 0.3536 along the normalized diagonal, i.e. (0.25, 0.25) in cell units
    /// for a 0.5-sized quad. The quad therefore spans from the tile's centre exactly
    /// to its corner. <see cref="CornerOffset"/> and <see cref="CornerQuadSize"/>
    /// carry those values.
    /// </summary>
    public class WallMeshBuilder
    {
        public const float CornerQuadSize = 0.5f;

        /// <summary>Offset of a corner patch's centre from the tile's centre, per axis.</summary>
        public const float CornerOffset = CornerQuadSize * 0.5f;

        /// <summary>
        /// How far corner patches sit in front of the wall quads, so they always win
        /// the depth test. RimWorld's Graphic_LinkedCornerFiller raises them by 0.09.
        /// </summary>
        public const float CornerDepthBias = 0.09f;

        private readonly List<Vector3> _verts = new List<Vector3>();
        private readonly List<Vector2> _uvs = new List<Vector2>();
        private readonly List<int> _tris = new List<int>();

        /// <summary>
        /// Build one mesh per wall material present on the map. Meshes already in
        /// <paramref name="meshes"/> are rewritten in place rather than reallocated,
        /// so this is safe to call on every wall change.
        /// </summary>
        public void Build(Map map, LinkFlags linkFlags, Dictionary<WallMaterial, Mesh> meshes)
        {
            for (int m = 0; m < WallMaterialDef.Count; m++)
            {
                var material = (WallMaterial)m;

                _verts.Clear();
                _uvs.Clear();
                _tris.Clear();

                BuildMaterial(map, material, linkFlags);

                if (!meshes.TryGetValue(material, out Mesh mesh) || mesh == null)
                {
                    mesh = new Mesh { name = "AgonWalls_" + material };
                    mesh.MarkDynamic();
                    meshes[material] = mesh;
                }

                mesh.Clear();

                if (_verts.Count == 0)
                    continue;

                // A map wider than 181 cells can exceed 65535 verts once corner
                // patches are counted, so index wide meshes with 32 bits.
                mesh.indexFormat = _verts.Count > 65000
                    ? UnityEngine.Rendering.IndexFormat.UInt32
                    : UnityEngine.Rendering.IndexFormat.UInt16;

                mesh.SetVertices(_verts);
                mesh.SetUVs(0, _uvs);
                mesh.SetTriangles(_tris, 0);
                mesh.RecalculateBounds();
            }
        }

        private void BuildMaterial(Map map, WallMaterial material, LinkFlags linkFlags)
        {
            for (int y = 0; y < map.Height; y++)
            {
                for (int x = 0; x < map.Width; x++)
                {
                    var cell = new Vector2Int(x, y);

                    if (!map.Edifices.IsWall(cell) || map.Edifices.MaterialAt(cell) != material)
                        continue;

                    LinkDirections links = map.Links.LinksAt(cell, linkFlags);
                    Rect uv = LinkedAtlas.UvRect(links);

                    AddQuad(new Vector2(x, y), Vector2.one, 0f, uv);
                    AddCornerFillers(map, cell, linkFlags);
                }
            }
        }

        /// <summary>
        /// Whether the tile at <paramref name="cell"/> draws a corner patch toward the
        /// diagonal at <paramref name="diagonalIndex"/> (an index into
        /// <see cref="Map.DiagonalDirections"/>).
        ///
        /// This is the whole of Graphic_LinkedCornerFiller's guard: the diagonal must
        /// link, and so must both cardinals flanking it. The flanking test is what
        /// keeps a patch from hanging off the end of a wall — a lone diagonal
        /// neighbour, as at the inside of an L, is not a corner to fill.
        /// </summary>
        public static bool ShouldFillCorner(Map map, Vector2Int cell, int diagonalIndex, LinkFlags linkFlags)
        {
            Vector2Int diagonal = Map.DiagonalDirections[diagonalIndex];

            if (!map.Links.ShouldLinkWith(cell + diagonal, linkFlags))
                return false;

            var flankA = new Vector2Int(diagonal.x, 0);
            var flankB = new Vector2Int(0, diagonal.y);

            return map.Links.ShouldLinkWith(cell + flankA, linkFlags)
                && map.Links.ShouldLinkWith(cell + flankB, linkFlags);
        }

        private void AddCornerFillers(Map map, Vector2Int cell, LinkFlags linkFlags)
        {
            Vector2 solid = LinkedAtlas.SolidUv();
            var solidUv = new Rect(solid.x, solid.y, 0f, 0f);

            for (int i = 0; i < 4; i++)
            {
                if (!ShouldFillCorner(map, cell, i, linkFlags))
                    continue;

                Vector2Int diagonal = Map.DiagonalDirections[i];

                var origin = new Vector2(
                    cell.x + (diagonal.x > 0 ? CornerQuadSize : 0f),
                    cell.y + (diagonal.y > 0 ? CornerQuadSize : 0f));

                AddQuad(origin, new Vector2(CornerQuadSize, CornerQuadSize), -CornerDepthBias, solidUv);
            }
        }

        /// <summary>
        /// Add a quad whose lower-left corner is <paramref name="origin"/>, in cell units.
        /// Negative <paramref name="z"/> is toward the camera, which looks down -Z.
        /// </summary>
        private void AddQuad(Vector2 origin, Vector2 size, float z, Rect uv)
        {
            int baseIndex = _verts.Count;

            _verts.Add(new Vector3(origin.x, origin.y, z));
            _verts.Add(new Vector3(origin.x + size.x, origin.y, z));
            _verts.Add(new Vector3(origin.x + size.x, origin.y + size.y, z));
            _verts.Add(new Vector3(origin.x, origin.y + size.y, z));

            _uvs.Add(new Vector2(uv.xMin, uv.yMin));
            _uvs.Add(new Vector2(uv.xMax, uv.yMin));
            _uvs.Add(new Vector2(uv.xMax, uv.yMax));
            _uvs.Add(new Vector2(uv.xMin, uv.yMax));

            _tris.Add(baseIndex);
            _tris.Add(baseIndex + 3);
            _tris.Add(baseIndex + 2);

            _tris.Add(baseIndex);
            _tris.Add(baseIndex + 2);
            _tris.Add(baseIndex + 1);
        }
    }
}
