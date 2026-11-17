using NUnit.Framework;
using UnityEngine;
using Agon.Core;
using Agon.Rendering;

namespace Agon.Tests
{
    /// <summary>
    /// Graphic_LinkedCornerFiller's guard, case by case. The rule it encodes:
    /// a corner is filled only where four linked tiles meet, so a 2x2 block fills at
    /// its centre, and nothing else does.
    ///
    /// DiagonalDirections order is SW, NW, NE, SE — index 2 is north-east.
    /// </summary>
    public class CornerFillerTests
    {
        private const LinkFlags Walls = LinkFlags.WallLike;

        private const int SW = 0, NW = 1, NE = 2, SE = 3;

        private static Map MakeMap()
        {
            var map = new Map(16, 16);
            map.Terrain.Fill(TerrainKind.Soil);
            return map;
        }

        private static void Wall(Map map, int x, int y) =>
            map.Edifices.Set(new Vector2Int(x, y), EdificeKind.Wall, WallMaterial.Timber);

        private static int FilledCorners(Map map, Vector2Int cell)
        {
            int n = 0;
            for (int i = 0; i < 4; i++)
                if (WallMeshBuilder.ShouldFillCorner(map, cell, i, Walls))
                    n++;

            return n;
        }

        [Test]
        public void TwoByTwoBlock_EachTileFillsTowardTheSharedCorner()
        {
            var map = MakeMap();

            // Block at (5,5)-(6,6). The shared corner is between all four.
            Wall(map, 5, 5); Wall(map, 6, 5);
            Wall(map, 5, 6); Wall(map, 6, 6);

            // The south-west tile's north-east diagonal is the shared corner.
            Assert.IsTrue(WallMeshBuilder.ShouldFillCorner(map, new Vector2Int(5, 5), NE, Walls));
            Assert.IsTrue(WallMeshBuilder.ShouldFillCorner(map, new Vector2Int(6, 5), NW, Walls));
            Assert.IsTrue(WallMeshBuilder.ShouldFillCorner(map, new Vector2Int(5, 6), SE, Walls));
            Assert.IsTrue(WallMeshBuilder.ShouldFillCorner(map, new Vector2Int(6, 6), SW, Walls));

            // Each tile fills exactly that one corner — the block's outer corners
            // have no diagonal neighbour and must stay unfilled.
            Assert.AreEqual(1, FilledCorners(map, new Vector2Int(5, 5)));
            Assert.AreEqual(1, FilledCorners(map, new Vector2Int(6, 6)));
        }

        [Test]
        public void StraightRun_FillsNothing()
        {
            var map = MakeMap();
            for (int x = 3; x <= 9; x++)
                Wall(map, x, 5);

            for (int x = 4; x <= 8; x++)
                Assert.AreEqual(0, FilledCorners(map, new Vector2Int(x, 5)),
                    "A straight wall has no corner to fill at x=" + x);
        }

        [Test]
        public void LCorner_FillsNothing_BecauseTheDiagonalIsEmpty()
        {
            var map = MakeMap();

            // An L: west arm and south arm meeting at (5,5). The inside diagonal
            // (4,4) is empty, so there is no four-tile meeting point.
            Wall(map, 5, 5);
            Wall(map, 4, 5);
            Wall(map, 5, 4);

            Assert.AreEqual(0, FilledCorners(map, new Vector2Int(5, 5)),
                "An L has a linked pair of cardinals but no linked diagonal.");
        }

        [Test]
        public void DiagonalTouch_WithoutFlankingCardinals_FillsNothing()
        {
            var map = MakeMap();

            // Two walls touching only at a corner. This is the case the flanking
            // guard exists for: without it, both would grow a patch into open ground.
            Wall(map, 5, 5);
            Wall(map, 6, 6);

            Assert.AreEqual(0, FilledCorners(map, new Vector2Int(5, 5)));
            Assert.AreEqual(0, FilledCorners(map, new Vector2Int(6, 6)));
        }

        [Test]
        public void ThickWall_FillsAlongItsSpine()
        {
            var map = MakeMap();

            // A wall two cells thick: rows 5 and 6, x from 3 to 8. Every interior
            // tile meets three others at a corner, so it fills toward that side.
            for (int x = 3; x <= 8; x++)
            {
                Wall(map, x, 5);
                Wall(map, x, 6);
            }

            // A tile in the middle of the bottom row fills both its upper corners.
            Assert.IsTrue(WallMeshBuilder.ShouldFillCorner(map, new Vector2Int(5, 5), NE, Walls));
            Assert.IsTrue(WallMeshBuilder.ShouldFillCorner(map, new Vector2Int(5, 5), NW, Walls));
            Assert.IsFalse(WallMeshBuilder.ShouldFillCorner(map, new Vector2Int(5, 5), SE, Walls),
                "Nothing below, so nothing to fill downward.");
        }

        [Test]
        public void DoorInAWall_StillCountsAsLinked_SoTheRunDoesNotBreak()
        {
            var map = MakeMap();

            // Two thick wall stubs with a door between them, all on two rows.
            Wall(map, 4, 5); Wall(map, 4, 6);
            map.Edifices.Set(new Vector2Int(5, 5), EdificeKind.Door);
            map.Edifices.Set(new Vector2Int(5, 6), EdificeKind.Door);
            Wall(map, 6, 5); Wall(map, 6, 6);

            Assert.IsTrue(WallMeshBuilder.ShouldFillCorner(map, new Vector2Int(4, 5), NE, Walls),
                "A door links like a wall, so the corner beside it is still a meeting point.");
        }

        [Test]
        public void MapCorner_FillsAgainstTheEdge()
        {
            var map = MakeMap();

            // A wall in the map's corner: off-map counts as linked on two sides and
            // diagonally, so the outward corner is filled rather than left open.
            Wall(map, 0, 0);

            Assert.IsTrue(WallMeshBuilder.ShouldFillCorner(map, new Vector2Int(0, 0), SW, Walls),
                "Off-map links, so the wall closes against the map edge.");
        }
    }
}
