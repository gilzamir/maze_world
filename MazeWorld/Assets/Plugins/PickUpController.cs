using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using bworld;
public class PickUpController : MonoBehaviour {

	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
        if (PlayerLogicScene.gameIsPaused)
        {
            return;
        }
        transform.Rotate(new Vector3(15, 30, 45) * Time.deltaTime);
	}
}
