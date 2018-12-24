using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using bworld;

public class PlayerController : MonoBehaviour
{

    public float speed;
    private Rigidbody rb;
    private int count;
    public Text countText;
    public Text winText;

    private static PlayerController instance;

    // Use this for initialization
    void Start()
    {
        instance = this;
        rb = GetComponent<Rigidbody>();
        Reset();
    }

    public static void Reset()
    {
        instance.count = 0;
        instance.SetCountText();
        instance.winText.text = "";
    }
    

    private void Update()
    {
        if (PlayerLogicScene.gameIsPaused)
        {
            return;
        }
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (PlayerLogicScene.gameIsPaused)
        {
            return;
        }

        Vector3 force = new Vector3(Input.GetAxis("Horizontal"), 0.0f, Input.GetAxis("Vertical"));

        rb.AddForce(force * speed);
    }

    void OnTriggerEnter(Collider other)
    {
        if (PlayerLogicScene.gameIsPaused)
        {
            return;
        }
        if (other.gameObject.CompareTag("PickUp"))
        {
            other.gameObject.SetActive(false);
            count = count + 1;
            SetCountText();
        }
    }

    void SetCountText()
    {
        countText.text = "Count: " + count.ToString();
        if (count >= 12)
        {
            winText.text = "Congratulations!!! You Win!!!";
        }
    }
}
