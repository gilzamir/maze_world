using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace bworld
{

    public class PlayerLogicScene : MonoBehaviour
    {

        public static bool gameIsPaused = false;
        private bool isDone = false;

        // Use this for initialization
        void Start()
        {
            gameIsPaused = false;
        }

        // Update is called once per frame
        void Update()
        {

        }

        virtual public int GetLifes()
        {
            return 0;
        }

        virtual public float GetEnergy()
        {
            return 0;
        }

        virtual public int GetScore()
        {
            return 0;
        }

        public bool IsDone()
        {
            return this.isDone;
        }

        public void SetDone(bool value)
        {
            this.isDone = value;
        }

        virtual public void releaseFlag()
        {
        }

        virtual public void getFlag()
        {

        }

        virtual public bool IsNearOfFlag()
        {
            return false;
        }

        virtual public void Restart()
        {
            
        }

        public virtual string GetSceneName()
        {
            return "";
        }

        public virtual bool IsNearOfPickUp()
        {
            return false;
        }

        public virtual int getNearPickUpValue()
        {
            return 0;
        }

        public virtual bool IsWithKey()
        {
            return false;
        }

        public virtual void getPickUp()
        {

        }

        public virtual void releasePickUp()
        {

        }
    }
}