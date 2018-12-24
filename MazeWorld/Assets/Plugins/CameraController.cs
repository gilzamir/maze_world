using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using bworld;
public class CameraController : MonoBehaviour {

    public GameObject player;

    private Vector3 offset;

	// Use this for initialization
	void Start () {
        offset = transform.position - player.transform.position;
	}
	
	// Update is called once per frame
	void LateUpdate () {
        if (PlayerLogicScene.gameIsPaused)
        {
            return;
        }
        transform.position = player.transform.position + offset;
        transform.rotation.eulerAngles.Set(player.transform.rotation.eulerAngles.x, transform.rotation.eulerAngles.y,
                                          player.transform.eulerAngles.z);
	}
}
