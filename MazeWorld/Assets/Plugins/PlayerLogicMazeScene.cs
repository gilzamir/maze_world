using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using bworld;

namespace bworld
{
    public class PlayerLogicMazeScene : PlayerLogicScene
    {

        private float elapsedTime = 0;

        private const int MISSION_ACCOMPLISHED = 200;
        private const int MISION_NON_ACCOMPLISHED = -200;
        private const int ENERGY_LOSS = 1;
        private const int TARGET_VALUE = 100;

        private Rigidbody rb;

        private int energy = 15;
        private int MAX_NUMBER_OF_FLAGS = 2000;
        private int count;
        private int countFlags;

        public Text countText;
        public Text winText;
        public Text energyText;
        public Text flagsText;

        private bool isWithKey;
        private bool isInTeletransporter;
        private bool isNearOfFlag;
        private bool isNearOfPickUp;

        private const string TROFEU_STATUS_DISPLAY_NAME = "KeyHolded";
        private const string FLAGS_STATUS_DISPLAY_NAME = "FlagDetected";
        private const string PICKUP_STATUS_DISPLAY_NAME = "PickUPDetected";

        private GameObject trofeuStatusDisplay;
        private GameObject flagDetectedStatus;
        private GameObject pickUpDetectedStatus;
        private MazeSceneLogic currentSceneLogic;
        private GameObject preLoader;

        private GameObject btnRestart;
        private GameObject nearFlag;
        private GameObject nearPickUp;
        private static PlayerLogicMazeScene instance;
        private int nearPickUpValue = 0;


        public GameObject flag;
        public Sprite redPickUp;
        public Sprite greenPickUp;
        public float flagDistance;

        override
        public int GetLifes()
        {
            return 0;
        }
        
        override
        public float GetEnergy()
        {
            return energy;
        }

        override
        public int GetScore()
        {
            return count;
        }


        // Use this for initialization
        void Start()
        {
            instance = this;
            Reset();
        }

        public static void Reset()
        {
            instance.isWithKey = false;
            instance.isInTeletransporter = false;
            instance.SetDone(false);
            instance.isNearOfFlag = false;
            instance.countFlags = instance.MAX_NUMBER_OF_FLAGS;
            instance.isNearOfPickUp = false;
            instance.nearPickUpValue = 0;

            instance.btnRestart = GameObject.Find("BtnRestart");
            instance.btnRestart.GetComponent<Button>().onClick.AddListener(instance.Restart);
            instance.btnRestart.SetActive(false);


            instance.preLoader = GameObject.Find("PreLoader");
            instance.currentSceneLogic = instance.preLoader.GetComponent<MazeSceneLogic>();

            instance.flagDetectedStatus = GameObject.Find(FLAGS_STATUS_DISPLAY_NAME);
            if (instance.flagDetectedStatus != null)
            {
                instance.flagDetectedStatus.SetActive(false);
            }
            instance.pickUpDetectedStatus = GameObject.Find(PICKUP_STATUS_DISPLAY_NAME);
            instance.pickUpDetectedStatus.SetActive(false);

            instance.trofeuStatusDisplay = GameObject.Find(TROFEU_STATUS_DISPLAY_NAME);
            instance.trofeuStatusDisplay.SetActive(false);
            instance.countText.text = "Score: 0";
            instance.winText.text = "";
            instance.energyText.text = "Energy: 1000";
            instance.flagsText.text = "Flags:" + instance.MAX_NUMBER_OF_FLAGS;
        }

        void OnTriggerStay(Collider other)
        {
            if (PlayerLogicScene.gameIsPaused)
            {
                return;
            }
            if (other.gameObject.CompareTag("PickUp") || other.gameObject.CompareTag("PickUpBad"))
            {
                int idx = int.Parse(other.gameObject.name);
                pickUpDetectedStatus.SetActive(true);
                nearPickUp = other.gameObject;

                Image img = pickUpDetectedStatus.GetComponent<Image>();
                if (currentSceneLogic.getPickUpReward()[idx] > 0)
                {
                    img.sprite = greenPickUp;
                    nearPickUpValue = 10;
                } else
                {
                    nearPickUpValue = -10;
                    img.sprite = redPickUp;
                }
                isNearOfPickUp = true;
            }
            else if (other.gameObject.CompareTag("Target"))
            {
                count = count + MISSION_ACCOMPLISHED;
                isWithKey = true;
                other.gameObject.SetActive(false);
                trofeuStatusDisplay.SetActive(true);
                count += TARGET_VALUE;
                SetCountText();
            }
            else if (other.gameObject.CompareTag("Teletransporter"))
            {
                isInTeletransporter = true;
                SetCountText();
            }
            else if (other.gameObject.CompareTag("flag"))
            {
                flagDetectedStatus.SetActive(true);
                nearFlag = other.gameObject;
                isNearOfFlag = true;
            }
        }

        public override bool IsWithKey()
        {
            return this.isWithKey;
        }

        override public void releaseFlag()
        {
            if (countFlags > 0)
            {
                Instantiate<GameObject>(flag, new Vector3(transform.position.x + transform.forward.x * flagDistance, -142, transform.position.z + transform.forward.z * flagDistance), flag.transform.rotation).name = "" + countFlags;
                countFlags--;
            }
        }

        public override bool IsNearOfPickUp()
        {
            return this.isNearOfPickUp;
        }

        public override int getNearPickUpValue()
        {
            return this.nearPickUpValue;
        }

        public override bool IsNearOfFlag()
        {
            return this.isNearOfFlag;
        }

        override public void getFlag()
        {
            if (isNearOfFlag)
            {
                countFlags++;
                nearFlag.SetActive(false);
                Destroy(nearFlag);
                nearFlag = null;
                flagDetectedStatus.SetActive(false);
                isNearOfFlag = false;
            }
        }

        public override void getPickUp()
        {
            if (isNearOfPickUp)
            {
                int idx = int.Parse(nearPickUp.name);
                nearPickUp.SetActive(false);
                Destroy(nearPickUp);
                nearPickUp = null;
                pickUpDetectedStatus.SetActive(false);
                isNearOfPickUp = false;
                energy = energy + currentSceneLogic.getPickUpReward()[idx];
                if (energy < 0) energy = 0;
                SetCountText();
            }
        }

        private void OnTriggerExit(Collider other)
        {
            if (PlayerLogicScene.gameIsPaused)
            {
                return;
            }

            if (other.gameObject.CompareTag("Teletransporter"))
            {
                isInTeletransporter = false;
            }
            else if (other.gameObject.CompareTag("flag"))
            {
                flagDetectedStatus.SetActive(false);
                isNearOfFlag = false;
                nearFlag = null;
            } else if (other.gameObject.CompareTag("PickUp") || other.gameObject.CompareTag("PickUpBad"))
            {
                pickUpDetectedStatus.SetActive(false);
                isNearOfPickUp = false;
                nearPickUpValue = 0;
                nearPickUp = null;
            }
        }

        void SetCountText()
        {
            if (energy > 0 && isWithKey && isInTeletransporter && !IsDone())
            {
                winText.text = " You Win!!!";
                btnRestart.SetActive(true);
                gameObject.transform.position = new Vector3(262.68f, -143.31f, 305.319f);
                SetDone(true);
                Time.timeScale = 0;
            }
            else if (energy <= 0 && !IsDone())
            {
                winText.text = "Game Over!";
                count = count + MISION_NON_ACCOMPLISHED;
                btnRestart.SetActive(true);
                gameObject.transform.position = new Vector3(262.68f, -143.31f, 305.319f);
                SetDone(true);
                Time.timeScale = 0;
            }
            countText.text = "Score: " + count;
            energyText.text = "Energy: " + energy;
            flagsText.text = "Flags: " + countFlags;
        }

        override
        public void Restart()
        {
            PlayerLogicScene.gameIsPaused = false;
            SetDone(false);
            Time.timeScale = 1;
            SceneManager.LoadScene("maze");
        }

        override
        public string GetSceneName()
        {
            return "maze";
        }

        
        // Update is called once per frame
        void Update()
        {
            if (PlayerLogicScene.gameIsPaused)
            {
                return;
            }

            elapsedTime += UnityEngine.Time.deltaTime;
            if (elapsedTime > 1.0f)
            {
                energy -= ENERGY_LOSS;
                if (energy < 0) energy = 0;
                elapsedTime = 0.0f;
                SetCountText();
            }

        }
    }
}