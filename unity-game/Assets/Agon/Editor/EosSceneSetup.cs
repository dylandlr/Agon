using Agon.Eos;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class EosSceneSetup
{
    [MenuItem("Agon/Create Eos scene")]
    public static void Create()
    {
        const string path = "Assets/Eos.unity";
        if (System.IO.File.Exists(path))
            throw new System.InvalidOperationException("Eos scene already exists; open it instead of replacing it.");
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        new GameObject("Eos authority client").AddComponent<EosClient>();
        EditorSceneManager.SaveScene(scene, path);
        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(path, true) };
        PlayerSettings.runInBackground = true;
        AssetDatabase.SaveAssets();
    }

    public static void AddHud()
    {
        var scene = EditorSceneManager.OpenScene("Assets/Eos.unity", OpenSceneMode.Single);
        var client = Object.FindAnyObjectByType<EosClient>();
        if (client == null) throw new System.InvalidOperationException("Eos client is missing");
        if (client.GetComponent<EosHud>() == null) client.gameObject.AddComponent<EosHud>();
        EditorSceneManager.SaveScene(scene);
    }
}
