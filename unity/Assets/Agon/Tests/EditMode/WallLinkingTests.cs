using NUnit.Framework;
using UnityEngine;
using Agon.Core;
using Agon.Rendering;

namespace Agon.Tests
{
    /// <summary>
    /// The linking rules, checked against RimWorld's own behaviour. These are the
    /// tests that say the walls are "correct": the bit order and atlas layout come
    /// from Verse.Graphic_Linked and Verse.MaterialAtlasPool, and the corner rule
    /// from Verse.Graphic_LinkedCornerFiller.
    /// </summary>
    public class WallLinkingTests
    {
        private const LinkFlags Walls = LinkFlags.WallLike;

        private static Map MakeMap(int w = 16, int h = 16)
        {
            var map = new Map(w, h);
            map.Terrain.Fill(TerrainKind.Soil);
            return map;
        }

        private static void Wall(Map map, int x, int y) =>
            map.Edifices.Set(new Vector2Int(x, y), EdificeKind.Wall, WallMaterial.Timber);

        [Test]
        public void LinkMask_BitOrder_MatchesRimWorld()
        {
            var map = MakeMap();
            var centre = new Vector2Int(5, 5);
            Wall(map, 5, 5);

            Assert.AreEqual(LinkDirections.None, map.Links.LinksAt(centre, Walls),
                "An isolated wall links to nothing.");

            Wall(map, 5, 6);
            Assert.AreEqual(LinkDirections.Up, map.Links.LinksAt(centre, Walls));
            Assert.AreEqual(1, (int)map.Links.LinksAt(centre, Walls), "Up must be bit 1.");

            Wall(map, 6, 5);
            Assert.AreEqual(3, (int)map.Links.LinksAt(centre, Walls), "Right must be bit 2.");

            Wall(map, 5, 4);
            Assert.AreEqual(7, (int)map.Links.LinksAt(centre, Walls), "Down must be bit 4.");

            Wall(map, 4, 5);
            Assert.AreEqual(15, (int)map.Links.LinksAt(centre, Walls), "Left must be bit 8.");
            Assert.AreEqual(LinkDirections.All, map.Links.LinksAt(centre, Walls));
        }

        [Test]
        public void LinkMask_LinksWithDoorsAndRock_NotWithEmptyGround()
        {
            var map = MakeMap();
            var centre = new Vector2Int(5, 5);
            Wall(map, 5, 5);

            map.Edifices.Set(new Vector2Int(5, 6), EdificeKind.Door);
            map.Edifices.Set(new Vector2Int(6, 5), EdificeKind.NaturalRock);

            LinkDirections links = map.Links.LinksAt(centre, Walls);

            Assert.IsTrue((links & LinkDirections.Up) != 0, "A wall links through a door.");
            Assert.IsTrue((links & LinkDirections.Right) != 0, "A wall links into natural rock.");
            Assert.IsTrue((links & LinkDirections.Down) == 0, "Empty ground does not link.");
            Assert.IsTrue((links & LinkDirections.Left) == 0, "Empty ground does not link.");
        }

        [Test]
        public void LinkMask_MapEdgeCounts_AsLinked()
        {
            var map = MakeMap();
            var corner = new Vector2Int(0, 0);
            Wall(map, 0, 0);

            LinkDirections links = map.Links.LinksAt(corner, Walls);

            Assert.IsTrue((links & LinkDirections.Down) != 0, "Off-map to the south links.");
            Assert.IsTrue((links & LinkDirections.Left) != 0, "Off-map to the west links.");
        }

        [Test]
        public void AtlasUv_MatchesRimWorldsAtlasMath()
        {
            // MaterialAtlasPool: offset = (i % 4 * 0.25 + 0.03125, i / 4 * 0.25 + 0.03125),
            // scale = 0.1875.
            for (int i = 0; i < LinkedAtlas.CellCount; i++)
            {
                Rect uv = LinkedAtlas.UvRect((LinkDirections)i);

                Assert.AreEqual((i % 4) * 0.25f + 0.03125f, uv.x, 1e-6f, "cell " + i + " x");
                Assert.AreEqual((i / 4) * 0.25f + 0.03125f, uv.y, 1e-6f, "cell " + i + " y");
                Assert.AreEqual(0.1875f, uv.width, 1e-6f);
                Assert.AreEqual(0.1875f, uv.height, 1e-6f);
            }
        }

        [Test]
        public void AtlasUv_CellsDoNotOverlap_AndStayInsideTheSheet()
        {
            for (int i = 0; i < LinkedAtlas.CellCount; i++)
            {
                Rect a = LinkedAtlas.UvRect((LinkDirections)i);

                Assert.GreaterOrEqual(a.xMin, 0f);
                Assert.GreaterOrEqual(a.yMin, 0f);
                Assert.LessOrEqual(a.xMax, 1f);
                Assert.LessOrEqual(a.yMax, 1f);

                for (int j = i + 1; j < LinkedAtlas.CellCount; j++)
                {
                    Rect b = LinkedAtlas.UvRect((LinkDirections)j);
                    Assert.IsFalse(a.Overlaps(b), "cells " + i + " and " + j + " overlap");
                }
            }
        }

        [Test]
        public void SolidUv_SitsInsideTheFullyLinkedCell()
        {
            Vector2 solid = LinkedAtlas.SolidUv();
            Rect full = LinkedAtlas.UvRect(LinkDirections.All);

            Assert.IsTrue(full.Contains(solid),
                "The corner patch must sample a cell that is solid wall everywhere.");
        }
    }
}
