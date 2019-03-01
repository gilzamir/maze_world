using UnityEngine;
using UnityEngine.UI;
using System.Net.Sockets;
using System.Net;
using System.Text;
using UnityStandardAssets.Characters.ThirdPerson;
using UnityEngine.SceneManagement;
namespace bworld
{

    public class PlayerRemoteAgent : MonoBehaviour
    {
        //BEGIN::Network connection variables
        private int port;
        private static UdpClient udpSocket;
        //END::

        //BEGIN::Game controller variables
        private ThirdPersonCharacter character;
        private Transform m_CamTransform;
        private Vector3 m_CamForward;             // The current forward direction of the camera
        private Vector3 m_Move;
        //END::

        //BEGIN::motor controll variables
        private static float fx, fy;
        public string remoteIPAddress;
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
        public static bool isToRestart;
        private static UdpClient socket;
        private static bool commandReceived;
        public int frameWidth;
        public int frameHeight;
        public int rayCastingWidth;
        public int rayCastingHeight;
        //END::

        private GameObject player;

        private PlayerLogicScene playerSceneLogic;
        private static PlayerRemoteAgent remoteController;
        private PlayerRemoteSensor sensor;

        public Camera m_camera;

        // Use this for initialization
        void Start()
        {
            
            commandReceived = false;
            port = ConfigureSceneScript.inputPort;
            remoteController = this;
            Reset();
            if (!gameObject.activeSelf)
            {
                return;
            }
            player = GameObject.FindGameObjectsWithTag("Player")[0];
            playerSceneLogic = player.GetComponent<PlayerLogicScene>();

            if (m_camera != null)
            {
                m_CamTransform = m_camera.transform;
            }
            else
            {
                Debug.LogWarning(
                    "Warning: no main camera found. Third person character needs a Camera tagged \"MainCamera\", for camera-relative controls.", gameObject);
                // we use self-relative controls in this case, which probably isn't what the user wants, but hey, we warned them!
            }

            // get the third person character ( this should never be null due to require component )
            character = GetComponent<ThirdPersonCharacter>();
            sensor = new PlayerRemoteSensor();
            sensor.Start(m_camera, player, this.rayCastingHeight, this.rayCastingWidth, remoteIPAddress);
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
                udpSocket.BeginReceive(new System.AsyncCallback(ReceiveData), udpSocket);

                //Debug.Log("Listening");
            }
            catch (System.Exception e)
            {
                // Something went wrong
                Debug.Log("Socket error: " + e);
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

        void OnRenderImage(RenderTexture source, RenderTexture destination)
        {
            Texture2D image = new Texture2D(source.width, source.height);
            image.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0);
            TextureScale.Bilinear(image, frameWidth, frameHeight);
            byte[] currentFrame = image.EncodeToJPG();

            Destroy(image);
            Graphics.Blit(source, destination);
        }


        public static void ReceiveData(System.IAsyncResult result)
        {
            socket = null;
            try
            {
                // Debug.Log("Command received");
                socket = result.AsyncState as UdpClient;
                IPEndPoint source = new IPEndPoint(0, 0);

                byte[] data = socket.EndReceive(result, ref source);

                crouch = false;
                jump = false;
                getpickup = false;
                walkspeed = false;
                fx = 0;
                fy = 0;
                speed = 0.0f;

                if (data != null)
                {
                    resetState = false;
                    string cmd = Encoding.UTF8.GetString(data);
                    if (!cmd.Equals("update_sensor"))
                    {
                        string[] tokens = cmd.Trim().Split(';');
                        string cmdtype = tokens[0].Trim();

                        bool isAction = false;
                        if (cmdtype.StartsWith("action"))
                        {
                            if (!cmdtype.Contains("ignore frame"))
                                commandReceived = true;
                            isAction = true;
                        }

                        if (tokens.Length > 1)
                        {
                            
                            if (isAction)
                                fx = float.Parse(tokens[1].Trim(), System.Globalization.CultureInfo.InvariantCulture.NumberFormat);

                            if (tokens.Length > 2 && isAction)
                            {
                                fy = float.Parse(tokens[2].Trim(), System.Globalization.CultureInfo.InvariantCulture.NumberFormat);
                            }

                            if (tokens.Length > 3 && isAction)
                            {
                                speed = float.Parse(tokens[3], System.Globalization.CultureInfo.InvariantCulture.NumberFormat);
                            }

                            if (tokens.Length > 4 && isAction)
                            {
                                crouch = bool.Parse(tokens[4].Trim());
                            }

                            if (tokens.Length > 5 && isAction)
                            {
                                jump = bool.Parse(tokens[5].Trim());
                            }

                            if (tokens.Length > 6 && isAction)
                            {
                                leftTurn = float.Parse(tokens[6].Trim(), System.Globalization.CultureInfo.InvariantCulture.NumberFormat);
                            }

                            if (tokens.Length > 7 && isAction)
                            {
                                rightTurn = float.Parse(tokens[7].Trim(), System.Globalization.CultureInfo.InvariantCulture.NumberFormat);
                            }

                            if (tokens.Length > 8 && isAction)
                            {
                                up = float.Parse(tokens[8].Trim(), System.Globalization.CultureInfo.InvariantCulture.NumberFormat);
                            }

                            if (tokens.Length > 9 && isAction)
                            {
                                down = float.Parse(tokens[9].Trim(), System.Globalization.CultureInfo.InvariantCulture.NumberFormat);
                            }

                            if (tokens.Length > 10 && isAction)
                            {
                                pushing = bool.Parse(tokens[10].Trim());
                            }

                            if (tokens.Length > 11)
                            {
                                resetState = bool.Parse(tokens[11].Trim());
                            }

                            if (tokens.Length > 12 && isAction)
                            {
                                getpickup = bool.Parse(tokens[12].Trim());
                            }

                            if (tokens.Length > 13 && isAction)
                            {
                                walkspeed = bool.Parse(tokens[13].Trim());
                            }

                            if (tokens.Length > 14)
                            {
                                bool restart = bool.Parse(tokens[14].Trim());
                                if (restart)
                                {
                                  
                                    remoteController.playerSceneLogic.SetDone(false);
                                    PlayerLogicScene.gameIsPaused = false;
                                    isToRestart = true;
                                }
                            }

                            if (tokens.Length > 15)
                            {
                                bool pause = bool.Parse(tokens[15].Trim());
                                if (pause)
                                {
                                    PlayerLogicScene.gameIsPaused = true;
                                }
                            }

                            if (tokens.Length > 16)
                            {
                                bool resume = bool.Parse(tokens[16].Trim());
                                if (resume)
                                {
                                    PlayerLogicScene.gameIsPaused = false;
                                }
                            }

                            if (tokens.Length > 17)
                            {
                                bool _close = bool.Parse(tokens[17].Trim());
                                if (_close)
                                {
                                    Application.Quit();
                                }
                            }
                        } 

                        socket.BeginReceive(new System.AsyncCallback(ReceiveData), udpSocket);
                    }
                    else
                    {
                        commandReceived = true;
                        socket.BeginReceive(new System.AsyncCallback(ReceiveData), udpSocket);
                    }
                }
            }
            catch (System.Exception e)
            {
                Debug.Log("Inexpected error: " + e.Message);
            }
        }


        void Update()
        {
            if (commandReceived)
            {
                //Debug.Log("command received");
                commandReceived = false;
                PlayerRemoteSensor.Update();
            }

            if (PlayerLogicScene.gameIsPaused)
            {
                if (Time.timeScale > 0)
                {
                    jump = false;
                    Time.timeScale = 0;
                }
            }
            else
            {
                if (Time.timeScale <= 0)
                {
                    Time.timeScale = 1;
                }

            }
            if (isToRestart)
            {
                if (socket != null)
                {
                    socket.Close();
                }

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
                return;
            }

            // read inputs
            float h = fx;
            float v = fy;


            // calculate move direction to pass to character
            if (m_CamTransform != null)
            {

                // calculate camera relative direction to move:
                m_CamForward = Vector3.Scale(m_CamTransform.forward, new Vector3(1, 0, 1)).normalized;
                m_Move = v * m_CamForward + h * m_CamTransform.right;

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
            character.Move(m_Move, crouch, jump, rightTurn - leftTurn, down - up, pushing);
            //character.Move(m_Move, crouch, m_Jump, h, v, pushing);
            jump = false;
        }
    }

    public class PlayerRemoteSensor
    {
        private string remoteIpAddress;
        private int remotePort = 8870;
        private byte[] currentFrame;
        public int frameWidth;
        public int frameHeight;

        private RenderTexture view;
        private Camera m_camera;

        private static Socket sock;
        private static IPAddress serverAddr;
        private static EndPoint endPoint;

        private GameObject player;
        private GameObject preLoader;

        private static PlayerLogicScene playerSceneLogic;
        private static int life, score;
        private static float energy;


        private int verticalResolution = 10;
        private int horizontalResolution = 10;
        private bool useRaycast = true;

        private static Ray[,] raysMatrix = null;
        private static int[,] viewMatrix = null;
        private Vector3 fw1 = new Vector3(), fw2 = new Vector3(), fw3 = new Vector3();

        private static PlayerRemoteSensor currentPlayer = null;


        public void SetCurrentFrame(byte[] cf)
        {
            this.currentFrame = cf;
        }

        // Use this for initialization
        public void Start(Camera cam, GameObject player, int rayCastingHRes, int rayCastingVRes, string remoteIpA)
        {
            this.remoteIpAddress = remoteIpA;

            this.verticalResolution = rayCastingVRes;
            this.horizontalResolution = rayCastingHRes;
            life = 0;
            score = 0;
            energy = 0;
            useRaycast = ConfigureSceneScript.useRayCasting;
            currentFrame = null;

            m_camera = cam;
            this.player = player;
            fw3 = m_camera.transform.forward;

            playerSceneLogic = player.GetComponent<PlayerLogicScene>();

            if (useRaycast)
            {
                if (raysMatrix == null)
                {
                    raysMatrix = new Ray[verticalResolution, horizontalResolution];
                }
                if (viewMatrix == null)
                {
                    viewMatrix = new int[verticalResolution, horizontalResolution];

                }
                for (int i = 0; i < verticalResolution; i++)
                {
                    for (int j = 0; j < horizontalResolution; j++)
                    {
                        raysMatrix[i, j] = new Ray();
                    }
                }
                currentFrame = updateCurrentRayCastingFrame();
            }

            remotePort = ConfigureSceneScript.outputPort;
            currentPlayer = this;

            if (sock == null)
            {
                sock = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
            }
            serverAddr = IPAddress.Parse(remoteIpAddress);
            endPoint = new IPEndPoint(serverAddr, remotePort);
        }


        public static void SendMessageFrom(object sender, byte[] image, bool forceDone = false)
        {
            if (currentPlayer == null) return;

            const int MSG_SIZE = 70;
            if (image == null)
            {
                image = currentPlayer.currentFrame;
            }

            Vector3 to = PlayerRemoteSensor.currentPlayer.m_camera.transform.forward;
            float cosang = Vector3.Dot(PlayerRemoteSensor.currentPlayer.fw3, to);
            float angulo = Mathf.Acos(cosang);

            Vector3 pos = PlayerRemoteSensor.currentPlayer.m_camera.transform.position;

            string msg = life + ";" + energy + ";" + score + ";" + (forceDone ? 1 : 0) + ";" +
                    (playerSceneLogic.IsNearOfPickUp() ? 1 : 0) + ";" + (playerSceneLogic.getNearPickUpValue()) 
                    + ";" + angulo + ";" + pos.x + ";" + pos.y + ";" +  pos.z + ";" + (playerSceneLogic.IsWithKey() ? 1 : 0);

            char[] cmsg = msg.ToCharArray();
            char[] fmsg = cmsg;
            if (cmsg.Length < MSG_SIZE)
            {
                fmsg = new char[MSG_SIZE];
                cmsg.CopyTo(fmsg, 0);
                for (int i = cmsg.Length; i < MSG_SIZE; i++)
                {
                    fmsg[i] = ' ';
                }
            }

            byte[] b = Encoding.UTF8.GetBytes(fmsg);

            byte[] msgc = new byte[b.Length + image.Length];
            b.CopyTo(msgc, 0);
            image.CopyTo(msgc, b.Length);
            sendData(msgc);
        }

        public static void Update()
        {
            byte[] frame = null;
            if (PlayerRemoteSensor.currentPlayer.useRaycast)
            {
                frame = PlayerRemoteSensor.currentPlayer.updateCurrentRayCastingFrame();
            }
            else if (PlayerRemoteSensor.currentPlayer.currentFrame != null)
            {
                frame = PlayerRemoteSensor.currentPlayer.currentFrame;
            }
            life = playerSceneLogic.GetLifes();
            energy = playerSceneLogic.GetEnergy();
            score = playerSceneLogic.GetScore();
            if (frame != null)
                SendMessageFrom(null, frame, playerSceneLogic.IsDone());
        }


        private byte[] updateCurrentRayCastingFrame()
        {
            UpdateRaysMatrix(m_camera.transform.position, m_camera.transform.forward, m_camera.transform.up, m_camera.transform.right);
            UpdateViewMatrix();
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < verticalResolution; i++)
            {
                for (int j = 0; j < horizontalResolution; j++)
                {
                    //Debug.DrawRay(raysMatrix[i, j].origin, raysMatrix[i, j].direction, Color.red);
                    sb.Append(viewMatrix[i, j]).Append(",");
                }
                sb.Append(";");
            }
            return Encoding.UTF8.GetBytes(sb.ToString().ToCharArray());
        }



        Texture2D RTImage(Camera cam)
        {
            RenderTexture currentRT = RenderTexture.active;
            RenderTexture.active = cam.targetTexture;
            cam.Render();
            Texture2D image = new Texture2D(cam.targetTexture.width, cam.targetTexture.height);
            image.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
            TextureScale.Bilinear(image, frameWidth, frameHeight);
            RenderTexture.active = currentRT;
            return image;
        }

        private void UpdateRaysMatrix(Vector3 position, Vector3 forward, Vector3 up, Vector3 right, float fieldOfView = 45.0f)
        {


            float vangle = 2 * fieldOfView / verticalResolution;
            float hangle = 2 * fieldOfView / horizontalResolution;

            float ivangle = -fieldOfView;

            for (int i = 0; i < verticalResolution; i++)
            {
                float ihangle = -fieldOfView;
                fw1 = (Quaternion.AngleAxis(ivangle + vangle * i, right) * forward).normalized;
                fw2.Set(fw1.x, fw1.y, fw1.z);

                for (int j = 0; j < horizontalResolution; j++)
                {
                    raysMatrix[i, j].origin = position;
                    raysMatrix[i, j].direction = (Quaternion.AngleAxis(ihangle + hangle * j, up) * fw2).normalized;
                }
            }


        }

        private void UpdateViewMatrix(float maxDistance = 500.0f)
        {
            for (int i = 0; i < verticalResolution; i++)
            {
                for (int j = 0; j < horizontalResolution; j++)
                {
                    RaycastHit hitinfo;
                    if (Physics.Raycast(raysMatrix[i, j], out hitinfo, maxDistance))
                    {
                        string objname = hitinfo.collider.gameObject.name;
                        switch (objname)
                        {
                            case "Terrain":
                                viewMatrix[i, j] = 2;
                                break;
                            case "maze":
                                viewMatrix[i, j] = 3;
                                break;
                            case "Teletransporter":
                                viewMatrix[i, j] = 4;
                                break;
                            case "GoldKey":
                                viewMatrix[i, j] = 5;
                                break;
                            default:
                                objname = hitinfo.collider.gameObject.tag;
                                if (objname.StartsWith("PickUpBad", System.StringComparison.CurrentCulture))
                                {
                                    viewMatrix[i, j] = 1;
                                }
                                else if (objname.StartsWith("PickUp", System.StringComparison.CurrentCulture))
                                {
                                    viewMatrix[i, j] = 6;
                                }
                                else
                                {
                                    viewMatrix[i, j] = -1;
                                }
                                break;
                        }
                    }
                    else
                    {
                        viewMatrix[i, j] = 0;
                    }
                }
            }
        }

        public static void sendData(byte[] data)
        {
            sock.SendTo(data, endPoint);
        }
    }
}