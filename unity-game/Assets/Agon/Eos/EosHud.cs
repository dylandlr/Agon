using UnityEngine;
using UnityEngine.UIElements;

namespace Agon.Eos
{
    [RequireComponent(typeof(EosClient))]
    public sealed class EosHud : MonoBehaviour
    {
        EosClient _client;
        VisualElement _root;
        PanelSettings _panel;
        int _selected = -1, _count = -1;

        void Start()
        {
            _client = GetComponent<EosClient>();
            _panel = ScriptableObject.CreateInstance<PanelSettings>();
            _panel.scaleMode = PanelScaleMode.ScaleWithScreenSize;
            _panel.referenceResolution = new Vector2Int(1440, 900);
            _panel.themeStyleSheet = Resources.Load<ThemeStyleSheet>("EosRuntimeTheme");
            var document = gameObject.AddComponent<UIDocument>();
            document.panelSettings = _panel;
            document.visualTreeAsset = Resources.Load<VisualTreeAsset>("EosHud");
            _root = document.rootVisualElement;
            _root.AddToClassList("eos-document");
            _root.style.unityFont = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            Bind("play", () => _client.Paused = !_client.Paused);
            Bind("step", () => { _client.Paused = true; _client.StepOnce(); });
            Bind("speed", () => _client.StepsPerRequest = _client.StepsPerRequest == 1 ? 4 : _client.StepsPerRequest == 4 ? 16 : 1);
            Bind("save", _client.SaveGame); Bind("load", _client.LoadGame);
            Bind("left", () => Pan(-2, 0)); Bind("right", () => Pan(2, 0));
            Bind("up", () => Pan(0, 2)); Bind("down", () => Pan(0, -2));
            Bind("zoom-in", () => Zoom(.8f)); Bind("zoom-out", () => Zoom(1.25f));
            Bind("home", () => {
                if (Camera.main == null || _client.Snapshot == null) return;
                int size = _client.Snapshot.grid_size;
                Camera.main.transform.position = new Vector3(size/2f, size/2f, -10);
                Camera.main.orthographicSize = size * .55f;
            });
        }

        void Bind(string name, System.Action action) => _root.Q<Button>(name).clicked += action;
        void Pan(float x, float y)
        {
            var camera = Camera.main;
            if (camera == null || _client.Snapshot == null) return;
            var p = camera.transform.position;
            p.x = Mathf.Clamp(p.x+x, 0, _client.Snapshot.grid_size);
            p.y = Mathf.Clamp(p.y+y, 0, _client.Snapshot.grid_size);
            camera.transform.position = p;
        }
        void Zoom(float factor)
        {
            if (Camera.main != null) Camera.main.orthographicSize = Mathf.Clamp(Camera.main.orthographicSize * factor, 3, 80);
        }
        void Update()
        {
            if (_root == null) return;
            var s = _client.Snapshot;
            _root.Q<Label>("status").text = _client.Failure ?? (_client.Busy ? "Waiting for Eos…" : _client.Status);
            _root.Q<Button>("play").text = _client.Paused ? "Run" : "Pause";
            _root.Q<Button>("speed").text = _client.StepsPerRequest + " ticks";
            foreach (var name in new[] { "step", "save", "load" }) _root.Q<Button>(name).SetEnabled(s != null && !_client.Busy);
            _root.Q<Button>("play").SetEnabled(s != null && _client.Failure == null);
            if (s == null) return;
            _root.Q<Label>("provenance").text = s.policy == "trained" ? "Learned Eos policy · " + s.engine_version + " · " + s.learning : "Diagnostic " + s.policy + " policy — not trained gameplay";
            _root.Q<Label>("clock").text = "Tick " + s.tick + "  /  " + s.tick_minutes + " minutes per tick";
            int visibleCount = 0;
            foreach (var agent in s.agents) if (agent.existed) visibleCount++;
            if (_count != visibleCount)
            {
                _count = visibleCount;
                var list = _root.Q<ScrollView>("agents"); list.Clear();
                foreach (var agent in s.agents)
                {
                    if (!agent.existed) continue;
                    int slot = agent.slot;
                    list.Add(new Button(() => _selected = slot) { text = "Agent " + (slot+1) });
                }
            }
            foreach (var a in s.agents) if (a.slot == _selected)
                _root.Q<Label>("details").text = "Agent " + (a.slot+1) + (a.alive ? "" : " — deceased") +
                    "\n\nFood  " + a.satiety.ToString("0.00") + "\nWater  " + a.hydration.ToString("0.00") +
                    "\nEnergy  " + a.energy.ToString("0.00") + "\nHealth  " + a.health.ToString("0.00") +
                    "\nStress  " + a.stress.ToString("0.00") + "\nInjury  " + a.injury.ToString("0.00") +
                    "\n\nCell  " + a.column + ", " + a.row + "\nAge  " + a.age_ticks + " ticks";
        }
        void OnDestroy() { if (_panel != null) Destroy(_panel); }
    }
}
