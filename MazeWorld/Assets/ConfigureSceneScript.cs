using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEngine.SceneManagement;
using UnityEngine.UI;


public class ConfigureSceneScript : MonoBehaviour
{


    public InputField inPort;
    public InputField outPort;

    public void OnMouseClick()
    {
        PlayerPrefs.SetInt("InputPort", int.Parse(inPort.text));
        PlayerPrefs.SetInt("OutputPort", int.Parse(outPort.text));
        SceneManager.LoadScene("maze");
    }

    // Start is called before the first frame upd

    public void Start()
    {
        
    }

    // Update is called once per fram

    public void Update()
    {
        
    }
}
