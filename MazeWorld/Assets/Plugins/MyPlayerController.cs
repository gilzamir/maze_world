using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MyPlayerController : MonoBehaviour {

	Animator animator;

	// Use this for initialization
	void Start () {
		animator = GetComponent<Animator>();
	}
	
	void jump () {
			animator.Play("jump");
			animator.speed = 1.0f;
	}

	// Update is called once per frame
	void Update () {
        float move = Input.GetAxis("Vertical");

        animator.SetFloat("Forward", move);

		if (Input.GetButtonDown("Jump"))
        {
            jump();
            Debug.Log(Input.mousePosition);
        }
		
	}
}
