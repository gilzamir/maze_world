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

        private static PlayerRemoteSensor currentPlayer = null;

        // Use this for initialization
        void Start()
        {
            life = 0;
            score = 0;
            energy = 0;
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


        public static void SendMessageFrom(object sender, byte []image, bool forceDone=false)
        {

            if (currentPlayer == null || !currentPlayer.isActiveAndEnabled) return;

            const int MSG_SIZE = 50;
            if (image == null)
            {
                image = currentPlayer.currentFrame;
            }

            string msg = life + ";" + energy + ";" + score + ";" + (forceDone ? 1 : 0) + ";" +
                    (playerSceneLogic.IsNearOfPickUp() ? 1 : 0) + ";" + (playerSceneLogic.getNearPickUpValue());

            char []cmsg = msg.ToCharArray();
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

        // Update is called once per frame
        void FixedUpdate()
        {
            if (PlayerLogicScene.gameIsPaused)
            {
                return;
            }

            if (currentFrame != null)
            {
                life = playerSceneLogic.GetLifes();
                energy = playerSceneLogic.GetEnergy();
                score = playerSceneLogic.GetScore();
                SendMessageFrom(null, currentFrame, playerSceneLogic.IsDone());
            }
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

        static void sendData(byte[] data)
        {
            sock.SendTo(data, endPoint);
        }
    }
}
