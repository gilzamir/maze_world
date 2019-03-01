using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using bworld;

public class MazeSceneLogic : MonoBehaviour {

    private const int TARGET_POSITIONS_LENGHT = 10;
    private const string TARGET_NAME = "GoldKey";
    private const int PICKUPMAP_SIZE = 80;
    private const float BAD_PICKUP_CHANCE = 0.4f;

    private float pickUpShift = -0.5f;
    private float targetPositionShifit = -2;

    private Vector3[] targetPositions = new Vector3[]{ new Vector3(262, -138.43f, 425), new Vector3(262, -138.43f, 208),
                                                       new Vector3(293f, -138.43f, 398f), new Vector3(293, -138.43f, 368f),
                                                       new Vector3(231, -138.43f, 368f), new Vector3(260, -138.43f, 464f),
                                                       new Vector3(350, -138.43f, 463), new Vector3(40, -138.43f, 172),
                                                       new Vector3(486, -138.43f, 241), new Vector3(486, -138.43f, 431f),
                                                       new Vector3(262, -138.43f, 172)};

    private float[] pickUpMap = new float[]
    {
        263, -138.5f, 347,
        263, -138.5f, 229,
        263, -138.5f, 295,
        263, -138.5f, 315,
        294, -138.5f, 238,
        325, -138.5f, 238,
        325, -138.5f, 208,
        294, -138.5f, 202,
        294, -138.5f, 80,
        263, -138.5f, 62,
        311, -138.5f, 46,
        355, -138.5f, 46,
        356, -138.5f, 16,
        387, -138.5f, 14,
        262, -138.5f, 14,
        357, -138.5f, 83,
        359, -138.5f, 138,
        389, -138.5f, 114,
        421, -138.5f, 170,
        453, -138.5f, 143,
        454, -138.5f, 48,
        484, -138.5f, 48,
        453, -138.5f, 105,
        390, -138.5f, 72,
        419, -138.5f, 43,
        484, -138.5f, 203,
        359, -138.5f, 208,
        357, -138.5f, 299,
        295, -138.5f, 331,
        356, -138.5f, 365,
        327, -138.5f, 395,
        357, -138.5f, 459,
        326, -138.5f, 459,
        260, -138.5f, 460,
        389, -138.5f, 460,
        421, -138.5f, 395,
        425, -138.5f, 366,
        452, -138.5f, 364,
        485, -138.5f, 335,
        485, -138.5f, 240,
        485, -138.5f, 395,
        454, -138.5f, 425,
        454, -138.5f, 237,
        391, -138.5f, 270,
        391, -138.5f, 335,
        357, -138.5f, 335,
        229, -138.5f, 207,
        197, -138.5f, 458,
        134, -138.5f, 458,
        40, -138.5f, 428,
        134, -138.5f, 430,
        135, -138.5f, 395,
        70, -138.5f, 395,
        38, -138.5f, 348,
        101, -138.5f, 348,
        198, -138.5f, 293,
        69, -138.5f, 293,
        101, -138.5f, 270,
        167, -138.5f, 253,
        231, -138.5f, 205,
        134, -138.5f, 207,
        40, -138.5f, 207,
        101, -138.5f,  176,
        197, -138.5f, 185,
        200, -138.5f, 176,
        99, -138.5f, 141,
        166, -138.5f, 141,
        229, -138.5f, 139,
        198, -138.5f, 82,
        38, -138.5f, 82,
        69, -138.5f, 75,
        198, -138.5f, 79,
        166, -138.5f, 137,
        166, -138.5f, 44,
        166, -138.5f, 13,
        70, -138.5f, 18,
        36, -138.5f, 18,
        43, -138.5f, 76,
        101, -138.5f, 16,
        197, -138.5f, 16,
    };

    private int[] pickup_reward = new int[PICKUPMAP_SIZE];

    public GameObject pickUpGood;
    public GameObject pickUpBad;
    
    private Vector3 backedVector3 = new Vector3();

    private static MazeSceneLogic instance; 

    public int[] getPickUpReward()
    {
        return pickup_reward;
    }

    // Use this for initialization
    void Start () {
        instance = this;
        Reset();
    }

    public static void Reset()
    {
        int targetPosition = ConfigureSceneScript.game_level;
        GameObject target = GameObject.Find(TARGET_NAME);
        instance.backedVector3.x = instance.targetPositions[targetPosition].x;
        instance.backedVector3.y = instance.targetPositions[targetPosition].y += instance.targetPositionShifit;
        instance.backedVector3.z = instance.targetPositions[targetPosition].z;
        target.transform.position = instance.backedVector3;

        for (int i = 0; i < PICKUPMAP_SIZE; i++)
        {

            int k = i * 3;
            float x = instance.pickUpMap[k];
            float y = instance.pickUpMap[k + 1] + instance.pickUpShift;
            float z = instance.pickUpMap[k + 2];
            float p = UnityEngine.Random.value;

            if (p <= BAD_PICKUP_CHANCE)
            {
                instance.pickup_reward[i] = -10;
                GameObject bgo = Instantiate<GameObject>(instance.pickUpBad, new Vector3(x, y, z), instance.pickUpBad.transform.rotation);
                bgo.name = "" + i;
                bgo.tag = "PickUpBad";
            }
            else
            {
                instance.pickup_reward[i] = 10;
                Instantiate<GameObject>(instance.pickUpGood, new Vector3(x, y, z), instance.pickUpGood.transform.rotation).name = "" + i;
            }
        }
    }
}
