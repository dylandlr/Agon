using System.Collections;
using Agon.Eos;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEngine.UIElements;

namespace Agon.Tests
{
    public class EosClientTests
    {
        GameObject _root;
        static void Press(VisualElement ui, string name)
        {
            var button = ui.Q<Button>(name);
            using (var submit = NavigationSubmitEvent.GetPooled())
            {
                submit.target = button;
                button.SendEvent(submit);
            }
        }
        [UnityTearDown] public IEnumerator Cleanup()
        {
            if (_root != null) Object.Destroy(_root);
            yield return null;
        }

        [UnityTest, Timeout(240000)]
        public IEnumerator RealEosProcessAdvancesAndPauses()
        {
            _root = new GameObject("Eos integration test");
            var client = _root.AddComponent<EosClient>();
            client.SaveDirectory = System.IO.Path.Combine(Application.temporaryCachePath, "AgonTests", System.Guid.NewGuid().ToString("N"));
            _root.AddComponent<EosHud>();
            client.Paused = true;
            float deadline = Time.realtimeSinceStartup + 180;
            while (client.Snapshot == null && client.Failure == null && Time.realtimeSinceStartup < deadline)
                yield return null;
            Assert.IsNull(client.Failure, client.Failure);
            Assert.IsNotNull(client.Snapshot, "Eos reset did not return");
            Assert.AreEqual("random", client.Snapshot.policy, "Diagnostic provenance must stay explicit");
            Assert.AreEqual(4, client.Snapshot.agents.Length);
            Assert.AreEqual(5, _root.GetComponentsInChildren<SpriteRenderer>().Length);
            int tick = client.Snapshot.tick;
            for (int i = 0; i < 10; i++) yield return null;
            Assert.AreEqual(tick, client.Snapshot.tick, "Paused client advanced");
            client.StepOnce();
            deadline = Time.realtimeSinceStartup + 180;
            while (client.Busy && client.Failure == null && Time.realtimeSinceStartup < deadline)
                yield return null;
            Assert.IsNull(client.Failure, client.Failure);
            Assert.AreEqual(tick + 1, client.Snapshot.tick, "Real Eos tick did not advance");
            Assert.AreEqual(24 * 24, client.Snapshot.fields.terrain.Length);
            var ui = _root.GetComponent<UIDocument>().rootVisualElement;
            Assert.IsNotNull(ui.Q<Button>("play"));
            Assert.IsNotNull(ui.Q<Button>("save"));
            Assert.IsNotNull(ui.Q<ScrollView>("agents"));
            Assert.AreEqual(FlexDirection.Row, ui.Q(className: "toolbar").resolvedStyle.flexDirection);
            Assert.Greater(ui.Q<Label>("status").resolvedStyle.color.r, 0.8f, "HUD theme was not applied");
            Assert.Greater(ui.worldBound.height, 300, "HUD must fill the game view");
            Assert.Less(ui.Q("details").worldBound.yMax, ui.Q("status").worldBound.yMin,
                "Inspector and status must not overlap");
            var roster = ui.Q<ScrollView>("agents");
            var added = new System.Collections.Generic.List<Button>();
            for (int i = 0; i < 60; i++) { var b = new Button { text = "Roster test " + i }; added.Add(b); roster.Add(b); }
            for (int i = 0; i < 5; i++) yield return null;
            Assert.Greater(roster.verticalScroller.highValue, 0, "Large rosters must scroll");
            roster.ScrollTo(added[added.Count - 1]);
            for (int i = 0; i < 5; i++) yield return null;
            Assert.LessOrEqual(added[added.Count - 1].worldBound.yMax, roster.contentViewport.worldBound.yMax + 2);
            foreach (var b in added) b.RemoveFromHierarchy();
            Press(ui, "save");
            deadline = Time.realtimeSinceStartup + 30;
            while (client.Busy && client.Failure == null && Time.realtimeSinceStartup < deadline) yield return null;
            Assert.IsNull(client.Failure, client.Failure);
            string saved = JsonUtility.ToJson(client.Snapshot);
            Press(ui, "step");
            deadline = Time.realtimeSinceStartup + 60;
            while (client.Busy && client.Failure == null && Time.realtimeSinceStartup < deadline) yield return null;
            Assert.IsNull(client.Failure, client.Failure);
            Assert.AreEqual(tick + 2, client.Snapshot.tick);
            Press(ui, "load");
            deadline = Time.realtimeSinceStartup + 30;
            while (client.Busy && client.Failure == null && Time.realtimeSinceStartup < deadline) yield return null;
            Assert.IsNull(client.Failure, client.Failure);
            Assert.AreEqual(saved, JsonUtility.ToJson(client.Snapshot), "Save/load changed Eos snapshot");
        }
    }
}
