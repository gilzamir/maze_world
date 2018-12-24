using UnityEngine;
using UnityEngine.UI;
using System.Net.Sockets;
using System.Net;
using System.Text;
using UnityStandardAssets.Characters.ThirdPerson;
using UnityEngine.SceneManagement;
namespace bworld
{

    public class PlayerRemoteController : MonoBehaviour
    {
        //BEGIN::Network connection variables
        public int port;
        private static UdpClient udpSocket;
        //END::

        //BEGIN::Game controller variables

        private ThirdPersonCharacter character;
        private Transform m_Cam;
        private Vector3 m_CamForward;             // The current forward direction of the camera
        private Vector3 m_Move;
        private bool m_Jump;
        //END::

        //BEGIN::motor controll variables
        private static float fx, fy;
        private static float speed = 0.0f;
        private static bool crouch;
        private static bool jump;
        private static float leftTurn = 0;
        private static float rightTurn = 0;
        private static float up = 0;
        private static float down = 0;
        private static bool pushing;
        private static bool resetState;
        private static bool getpickup;
        private static bool walkspeed;
        private static bool isToRestart;
        //END::

        private GameObject player;

        private PlayerLogicScene playerSceneLogic;
        private static PlayerRemoteController remoteController;


        // Use this for initialization
        void Start()
        {
            remoteController = this;
            Reset();
            port = PlayerPrefs.GetInt("InputPort");
            if (!gameObject.activeSelf)
            {
                return;
            }
            player = GameObject.FindGameObjectsWithTag("Player")[0];
            playerSceneLogic = player.GetComponent<PlayerLogicScene>();
            // get the transform of the main camera
            if (Camera.main != null)
            {
                m_Cam = Camera.main.transform;
            }
            else
            {
                Debug.LogWarning(
                    "Warning: no main camera found. Third person character needs a Camera tagged \"MainCamera\", for camera-relative controls.", gameObject);
                // we use self-relative controls in this case, which probably isn't what the user wants, but hey, we warned them!
            }
            // get the third person character ( this should never be null due to require component )
            character = GetComponent<ThirdPersonCharacter>();
        }

        private static void ResetState()
        {
            isToRestart = false;
            speed = 0.0f;

            fx = 0;
            fy = 0;
            crouch = false;
            jump = false;
            pushing = false;
            leftTurn = 0;
            rightTurn = 0;
            up = 0;
            down = 0;
        }

        public static void Reset()
        {
            isToRestart = false;
            speed = 0.0f;

            fx = 0;
            fy = 0;
            crouch = false;
            jump = false;
            pushing = false;
            leftTurn = 0;
            rightTurn = 0;
            up = 0;
            down = 0;
            try
            {

                udpSocket = new UdpClient(remoteController.port);
                udpSocket.BeginReceive(new System.AsyncCallback(receiveData), udpSocket);

                //Debug.Log("Listening");
            }
            catch (System.Exception e)
            {
                // Something went wrong
                //Debug.Log("Socket error: " + e);
            }
        }

        void OnDisable()
        {
            if (udpSocket != null)
            {
                //Debug.Log("Socket is closed...");
                udpSocket.Close();
            }
        }

        public static void receiveData(System.IAsyncResult result)
        {
            UdpClient socket = null;
            try
            {   
            // Debug.Log("Command received");
                socket = result.AsyncState as UdpClient;
                IPEndPoint source = new IPEndPoint(0, 0);

                byte[] data = socket.EndReceive(result, ref source);

                if (data != null)
                {
                    fx = 0;
                    fy = 0;
                    speed = 0.0f;
                    crouch = false;
                    jump = false;
                    getpickup = false;
                    resetState = false;
                    walkspeed = false;

                    string cmd = Encoding.UTF8.GetString(data);
                    if (cmd != "end")
                    {
                        //Debug.Log("Command received...");
                        string[] tokens = cmd.Trim().Split(';');
                        
                        if (tokens.Length > 0)
                        {

                            fx = float.Parse(tokens[0].Trim());

                            if (tokens.Length > 1)
                            {
                                fy = float.Parse(tokens[1].Trim());
                            }

                            if (tokens.Length > 2)
                            {
                                speed = float.Parse(tokens[2]);
                            }

                            if (tokens.Length > 3)
                            {
                                crouch = bool.Parse(tokens[3].Trim());
                            }

                            if (tokens.Length > 4)
                            {
                                jump = bool.Parse(tokens[4].Trim());
                            }

                            if (tokens.Length > 5)
                            {
                                leftTurn = float.Parse(tokens[5].Trim());
                            }

                            if (tokens.Length > 6)
                            {
                                rightTurn = float.Parse(tokens[6].Trim());
                            }

                            if (tokens.Length > 7)
                            {
                                up = float.Parse(tokens[7].Trim());
                            }

                            if (tokens.Length > 8)
                            {
                                down = float.Parse(tokens[8].Trim());
                            }

                            if (tokens.Length > 9)
                            {
                                pushing = bool.Parse(tokens[9].Trim());
                            }

                            if (tokens.Length > 10)
                            {
                                resetState = bool.Parse(tokens[10].Trim());
                            }

                            if (tokens.Length > 11)
                            {
                                getpickup = bool.Parse(tokens[11].Trim());
                            }

                            if (tokens.Length > 12)
                            {
                                walkspeed = bool.Parse(tokens[12].Trim());
                            }

                            if (tokens.Length > 13)
                            {
                                bool restart = bool.Parse(tokens[13].Trim());
                                if (restart)
                                {
                                    isToRestart = true;
                                }
                            }

                            if (tokens.Length > 14)
                            {
                                bool pause = bool.Parse(tokens[14].Trim());
                                if (pause)
                                {
                                    PlayerLogicScene.gameIsPaused = true;
                                }
                            }

                            if (tokens.Length > 15)
                            {
                                bool resume = bool.Parse(tokens[15].Trim());
                                if (resume)
                                {
                                    PlayerLogicScene.gameIsPaused = false;
                                }
                            }
                        }
                        socket.BeginReceive(new System.AsyncCallback(receiveData), udpSocket);
                    }
                    //Debug.Log("Command executed...");
                }
            }
            catch (System.Exception e)
            {
                //Debug.Log("Inexpected error: " + e.Message);
                //Debug.Log("Detail: " + e.StackTrace);
            }
        }


        void Update()
        {

            if (PlayerLogicScene.gameIsPaused)
            {
                if (Time.timeScale > 0)
                {
                    Time.timeScale = 1;
                }
            } else
            {
                if (Time.timeScale <= 0)
                {
                    Time.timeScale = 1;
                }
            }
            if (isToRestart)
            {
                PlayerLogicScene.gameIsPaused = false;
                Time.timeScale = 1;
                SceneManager.LoadScene(remoteController.playerSceneLogic.GetSceneName());  
            }
        }

        // Update is called once per frame
        void FixedUpdate()
        {
            if (PlayerLogicScene.gameIsPaused)
            {
                if (Time.timeScale > 0)
                {
                    Time.timeScale = 0;
                }
                return;
            }
            else
            {
                if (Time.timeScale <= 0)
                {
                    Time.timeScale = 1;
                }
            }
            // read inputs
            float h = fx;
            float v = fy;


            // calculate move direction to pass to character
            if (m_Cam != null)
            {

                // calculate camera relative direction to move:
                m_CamForward = Vector3.Scale(m_Cam.forward, new Vector3(1, 0, 1)).normalized;
                m_Move = v * m_CamForward + h * m_Cam.right;

            }
            else
            {
                // we use world-relative directions in the case of no main camera
                m_Move = v * Vector3.forward + h * Vector3.right;
            }


            // walk speed multiplier
            if (walkspeed) m_Move *= speed;
            
            if (resetState)
            {
                ResetState();
            }

            if (getpickup)
            {
                playerSceneLogic.getPickUp();
            }


            // pass all parameters to the character control script
            character.Move(m_Move, crouch, m_Jump, rightTurn - leftTurn,  down - up, pushing);
            //character.Move(m_Move, crouch, m_Jump, h, v, pushing);
            m_Jump = false;
        }
    }
}