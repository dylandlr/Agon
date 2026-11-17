using NUnit.Framework;
using UnityEngine;
using Agon.Core;

namespace Agon.Tests
{
    /// <summary>
    /// What a pawn decides to do, and whether it actually gets there and does it.
    /// These are the tests that say the Unity build is a simulation rather than a
    /// renderer: a hungry pawn beside a full store should end up fed.
    /// </summary>
    public class JobTests
    {
        private static Map OpenMap(int w = 20, int h = 20)
        {
            var map = new Map(w, h);
            map.Terrain.Fill(TerrainKind.Soil);
            return map;
        }

        private static Pawn PawnAt(Map map, int x, int y, string id = "A01")
        {
            var pawn = new Pawn(id, map, new Vector2Int(x, y));
            pawn.Needs.Food.CurLevel = 1f;
            pawn.Needs.Water.CurLevel = 1f;
            pawn.Needs.Rest.CurLevel = 1f;
            return pawn;
        }

        [Test]
        public void NothingPressing_MeansTheyWait()
        {
            Map map = OpenMap();
            Pawn pawn = PawnAt(map, 5, 5);

            Job job = JobGiver.GiveJob(pawn);

            Assert.AreEqual(JobDef.Wait, job.Def, "Full needs, nothing to do.");
        }

        [Test]
        public void Hunger_SendsThemToTheNearestStore()
        {
            Map map = OpenMap();
            map.Things.Add(ThingKind.FoodStore, new Vector2Int(15, 15), 10f);
            Thing near = map.Things.Add(ThingKind.FoodStore, new Vector2Int(6, 5), 10f);

            Pawn pawn = PawnAt(map, 5, 5);
            pawn.Needs.Food.CurLevel = 0.1f;

            Job job = JobGiver.GiveJob(pawn);

            Assert.AreEqual(JobDef.Eat, job.Def);
            Assert.AreSame(near, job.TargetThing, "The nearer store, not the first one added.");
        }

        [Test]
        public void TheWorstNeedWins_EvenWhenAnotherIsAlsoLow()
        {
            Map map = OpenMap();
            map.Things.Add(ThingKind.FoodStore, new Vector2Int(6, 5), 10f);
            map.Things.Add(ThingKind.WaterSource, new Vector2Int(4, 5), 10f);

            Pawn pawn = PawnAt(map, 5, 5);
            pawn.Needs.Food.CurLevel = 0.39f;   // just under its 0.40 threshold
            pawn.Needs.Water.CurLevel = 0.05f;  // far under its 0.45 threshold

            Assert.AreEqual(JobDef.Drink, JobGiver.GiveJob(pawn).Def,
                "Dying of thirst beats being peckish.");
        }

        [Test]
        public void AStoreAnotherPawnHasClaimed_IsNotHandedOutTwice()
        {
            Map map = OpenMap();
            Thing only = map.Things.Add(ThingKind.FoodStore, new Vector2Int(6, 5), 10f);

            Pawn first = PawnAt(map, 5, 5, "A01");
            Pawn second = PawnAt(map, 5, 6, "A02");
            first.Needs.Food.CurLevel = 0.1f;
            second.Needs.Food.CurLevel = 0.1f;

            Job firstJob = JobGiver.GiveJob(first);
            Assert.AreEqual(JobDef.Eat, firstJob.Def);
            Assert.AreSame(first, only.ReservedBy);

            Job secondJob = JobGiver.GiveJob(second);
            Assert.AreNotEqual(JobDef.Eat, secondJob.Def,
                "The only store is taken, so the second pawn must not be sent to it.");
        }

        [Test]
        public void AnEmptyStore_IsIgnored()
        {
            Map map = OpenMap();
            map.Things.Add(ThingKind.FoodStore, new Vector2Int(6, 5), 0f);

            Pawn pawn = PawnAt(map, 5, 5);
            pawn.Needs.Food.CurLevel = 0.1f;

            Assert.AreNotEqual(JobDef.Eat, JobGiver.GiveJob(pawn).Def);
        }

        [Test]
        public void NoStoredWater_MeansDrinkingFromTheLake()
        {
            Map map = OpenMap();
            map.Terrain.Set(new Vector2Int(10, 5), TerrainKind.Water);

            Pawn pawn = PawnAt(map, 5, 5);
            pawn.Needs.Water.CurLevel = 0.1f;

            Job job = JobGiver.GiveJob(pawn);

            Assert.AreEqual(JobDef.Drink, job.Def);
            Assert.IsNull(job.TargetThing, "Drinking from open water uses no stored thing.");
            Assert.IsTrue(map.Terrain.Walkable(job.TargetCell),
                "They stand on the bank, not in the water.");
        }

        [Test]
        public void HungryPawn_WalksToTheStoreAndEats()
        {
            Map map = OpenMap();
            Thing store = map.Things.Add(ThingKind.FoodStore, new Vector2Int(12, 5), 10f);

            Pawn pawn = PawnAt(map, 2, 5);
            pawn.Needs.Food.CurLevel = 0.1f;

            float before = pawn.Needs.Food.CurLevel;
            float stockBefore = store.Stock;

            for (int i = 0; i < 200 && pawn.Jobs.JobsDone == 0; i++)
                pawn.Tick();

            Assert.Greater(pawn.Jobs.JobsDone, 0, "The job has to finish within 200 ticks.");
            Assert.Greater(pawn.Needs.Food.CurLevel, before, "They should be less hungry for it.");
            Assert.Less(store.Stock, stockBefore, "And the store should be down by what they ate.");
            Assert.LessOrEqual(PathFinder.Heuristic(pawn.Position, store.Position),
                PathFinder.DiagonalCost + 0.01f, "They end up at the store.");
        }

        [Test]
        public void TiredPawn_SleepsInABed_AndWakesRested()
        {
            Map map = OpenMap();
            map.Things.Add(ThingKind.Bed, new Vector2Int(7, 5));

            Pawn pawn = PawnAt(map, 5, 5);
            pawn.Needs.Rest.CurLevel = 0.05f;

            // run until the job itself ends, not until rest crosses a line: stopping at
            // a rest level would catch the pawn mid-sleep and the waking check below
            // would fire while it was still in bed
            for (int i = 0; i < 400 && pawn.Jobs.JobsDone == 0; i++)
                pawn.Tick();

            Assert.Greater(pawn.Jobs.JobsDone, 0, "The sleep job has to finish.");
            Assert.Greater(pawn.Needs.Rest.CurLevel, 0.9f, "A night in a bed should restore them.");
            Assert.IsFalse(pawn.Asleep, "And they get up afterwards.");
        }

        [Test]
        public void AnUnreachableStore_FailsTheJobInsteadOfHangingForever()
        {
            Map map = OpenMap();

            // wall the store into a sealed box
            for (int x = 9; x <= 13; x++)
            {
                map.Edifices.Set(new Vector2Int(x, 3), EdificeKind.Wall, WallMaterial.Stone);
                map.Edifices.Set(new Vector2Int(x, 7), EdificeKind.Wall, WallMaterial.Stone);
            }
            for (int y = 3; y <= 7; y++)
            {
                map.Edifices.Set(new Vector2Int(9, y), EdificeKind.Wall, WallMaterial.Stone);
                map.Edifices.Set(new Vector2Int(13, y), EdificeKind.Wall, WallMaterial.Stone);
            }

            map.Things.Add(ThingKind.FoodStore, new Vector2Int(11, 5), 10f);

            Pawn pawn = PawnAt(map, 2, 5);
            pawn.Needs.Food.CurLevel = 0.1f;

            for (int i = 0; i < 50; i++)
                pawn.Tick();

            Assert.IsNotNull(pawn.Jobs.LastFailed, "The unreachable job must be recorded as failed.");
            Assert.AreEqual(new Vector2Int(2, 5), pawn.Position,
                "And the pawn should not have wandered off toward it.");
        }

        [Test]
        public void DeadPawn_ReleasesWhatItHadClaimed()
        {
            Map map = OpenMap();
            Thing bed = map.Things.Add(ThingKind.Bed, new Vector2Int(6, 5));

            Pawn pawn = PawnAt(map, 5, 5);
            pawn.Needs.Rest.CurLevel = 0.05f;
            pawn.Tick();

            Assert.AreSame(pawn, bed.ReservedBy, "They claimed the bed.");

            pawn.Needs.Food.CurLevel = 0f;
            pawn.Needs.Water.CurLevel = 0f;
            for (int i = 0; i < 1000 && pawn.Alive; i++)
                pawn.Tick();

            Assert.IsFalse(pawn.Alive);
            Assert.IsNull(bed.ReservedBy, "A dead pawn must not hold a bed for ever.");
        }
    }

    public class PathFollowerTests
    {
        [Test]
        public void Walking_TakesLongerOverRoughGround()
        {
            int TicksToCross(TerrainKind terrain)
            {
                var map = new Map(12, 3);
                map.Terrain.Fill(terrain);

                var pawn = new Pawn("A01", map, new Vector2Int(0, 1));
                Assert.IsTrue(pawn.Path.StartPath(new Vector2Int(8, 1)));

                int ticks = 0;
                while (pawn.Path.HasPath && ticks < 500) { pawn.Path.Tick(); ticks++; }

                Assert.AreEqual(new Vector2Int(8, 1), pawn.Position);
                return ticks;
            }

            int onSoil = TicksToCross(TerrainKind.Soil);
            int onForest = TicksToCross(TerrainKind.Forest);

            Assert.Greater(onForest, onSoil, "Forest costs 1.6 a cell, so it must take longer.");
        }

        [Test]
        public void AWallRaisedAcrossThePath_ForcesAReroute()
        {
            var map = new Map(14, 7);
            map.Terrain.Fill(TerrainKind.Soil);

            var pawn = new Pawn("A01", map, new Vector2Int(1, 3));
            Assert.IsTrue(pawn.Path.StartPath(new Vector2Int(12, 3)));

            for (int i = 0; i < 6; i++)
                pawn.Path.Tick();

            // wall off the straight line ahead of them
            for (int y = 0; y < 7; y++)
                if (y != 6) map.Edifices.Set(new Vector2Int(8, y), EdificeKind.Wall, WallMaterial.Stone);

            int ticks = 0;
            while (pawn.Path.HasPath && ticks < 600) { pawn.Path.Tick(); ticks++; }

            Assert.AreEqual(new Vector2Int(12, 3), pawn.Position,
                "They should find the gap rather than stop at the new wall.");
        }
    }
}
