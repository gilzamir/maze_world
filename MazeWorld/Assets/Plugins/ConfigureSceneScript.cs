using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEngine.SceneManagement;
using UnityEngine.UI;


public class ConfigureSceneScript : MonoBehaviour
{


    public InputField inPort;
    public InputField outPort;
    public InputField inGameLevel;
    public Toggle useRayCast;
    private bool noconfig = false;
    public static int inputPort = 8870;
    public static int outputPort = 8890;
    public static bool useRayCasting;
    public static int game_level = 0;
    public void OnMouseClick()
    {
        inputPort  = int.Parse(inPort.text);
        outputPort = int.Parse(outPort.text);
        game_level = int.Parse(inGameLevel.text);
        useRayCasting = useRayCast.isOn;
        SceneManager.LoadScene("maze");
    }

    // Start is called before the first frame upd

    public void Start()
    {
        try
        {
            string[] args = System.Environment.GetCommandLineArgs();
            noconfig = false;
            useRayCasting = true;
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == "--input_port")
                {
                    inputPort = int.Parse(args[i + 1]);
                }

                if (args[i] == "--output_port")
                {
                    outputPort = int.Parse(args[i + 1]);
                }

                if (args[i] == "--use_frame")
                {
                    useRayCasting = false;
                }

                if (args[i] == "--game_level")
                {
                    game_level = int.Parse(args[i + 1]);
                }

                noconfig |= args[i] == "--noconfig";
            }
            if (args.Length > 0)
            {
                if (args[args.Length-1] == "--noconfig")
                {
                    noconfig = true;
                }
                if (args[args.Length-1] == "--use_frame")
                {
                    useRayCasting = false;
                }
            }
            inPort.text = inputPort.ToString();
            outPort.text = outputPort.ToString();
            inGameLevel.text = game_level.ToString();
            if (useRayCasting)
            {
                useRayCast.isOn = true;
            }
            else
            {
                useRayCast.isOn = false;
            }

            if (noconfig)
            {
                SceneManager.LoadScene("maze");
            }
        } catch(System.Exception e) {
            Debug.Log("Invalid command line arguments: " + e.Message);
        }

    }

    // Update is called once per fram

    public void Update()
    {
        
    }
}
