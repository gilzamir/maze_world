using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Net;
using System.Text;

namespace bworld
{

    public class PlayerRemoteSensor : MonoBehaviour
    {
        public string remoteIpAddress;
        public int remotePort;
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


        public int verticalResolution = 10;
        public  int horizontalResolution = 10;
        public bool useRaycast = true;

        private static Ray[,] raysMatrix = null;
        private static int[,] viewMatrix = null;
        private Vector3 fw1 = new Vector3(), fw2 = new Vector3();

        private static PlayerRemoteSensor currentPlayer = null;

        // Use this for initialization
        void Start()
        {
            life = 0;
            score = 0;
            energy = 0;
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
                    raysMatrix[i,j] = new Ray();
                }
            }

            remotePort = PlayerPrefs.GetInt("OutputPort");
            currentPlayer = this;
            if (!gameObject.activeSelf)
            {
                return;
            }
            currentFrame = null;

            m_camera = GetComponent<Camera>();
            player = GameObject.FindGameObjectsWithTag("Player")[0];

            playerSceneLogic = player.GetComponent<PlayerLogicScene>();

            if (sock == null)
            {
                sock = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
            }
            serverAddr = IPAddress.Parse(remoteIpAddress);
            endPoint = new IPEndPoint(serverAddr, remotePort);
        }


        public static void SendMessageFrom(object sender, byte[] image, bool forceDone = false)
        {

            if (currentPlayer == null || !currentPlayer.isActiveAndEnabled) return;

            const int MSG_SIZE = 50;
            if (image == null)
            {
                image = currentPlayer.currentFrame;
            }

            string msg = life + ";" + energy + ";" + score + ";" + (forceDone ? 1 : 0) + ";" +
                    (playerSceneLogic.IsNearOfPickUp() ? 1 : 0) + ";" + (playerSceneLogic.getNearPickUpValue());

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


        public void Update()
        {
            /*
            if (Input.GetKey(KeyCode.P))
            {
                UpdateRaysMatrix(m_camera.transform.position, m_camera.transform.forward, m_camera.transform.up, m_camera.transform.right);
                UpdateViewMatrix();
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < verticalResolution; i++)
                {
                    for (int j = 0; j < horizontalResolution; j++)
                    {
                        Debug.DrawRay(raysMatrix[i,j].origin, raysMatrix[i,j].direction, Color.red);
                        sb.Append(viewMatrix[i, j]).Append(" ");
                    }
                    sb.Append("\n");
                }
                Debug.Log(sb.ToString());
            }*/
        }

        // Update is called once per frame
        void FixedUpdate()
        {
            if (PlayerLogicScene.gameIsPaused)
            {
                return;
            }
            byte[] frame = null;
            if (useRaycast)
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
                frame = Encoding.UTF8.GetBytes(sb.ToString().ToCharArray());
            }
            else  if (currentFrame != null)
            {
                frame = currentFrame;
            }
            life = playerSceneLogic.GetLifes();
            energy = playerSceneLogic.GetEnergy();
            score = playerSceneLogic.GetScore();
            SendMessageFrom(null, frame, playerSceneLogic.IsDone());
        }

        void OnRenderImage(RenderTexture source, RenderTexture destination)
        {
            Texture2D image = new Texture2D(source.width, source.height);
            image.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0);
            TextureScale.Bilinear(image, frameWidth, frameHeight);
            currentFrame = image.EncodeToJPG();
            Destroy(image);
            Graphics.Blit(source, destination);
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

        private void UpdateRaysMatrix(Vector3 position, Vector3 forward, Vector3 up, Vector3 right, float fieldOfView=90.0f)
        {


            float vangle = 2 * fieldOfView/verticalResolution;
            float hangle = 2 *  fieldOfView/horizontalResolution;

            float ivangle = -fieldOfView;

            for (int i = 0; i < verticalResolution; i++)
            {
                float ihangle = -fieldOfView;
                fw1 = (Quaternion.AngleAxis(ivangle+vangle*i, right)*forward).normalized;
                fw2.Set(fw1.x, fw1.y, fw1.z);

                for (int j = 0; j < horizontalResolution; j++)
                {
                    raysMatrix[i, j].origin = position;
                    raysMatrix[i, j].direction = (Quaternion.AngleAxis(ihangle+hangle*j, up) *  fw2).normalized;
                }
            }


        }

        private void UpdateViewMatrix(float maxDistance=500.0f)
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
                                if (objname.StartsWith("PickUpRed", System.StringComparison.CurrentCulture))
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
