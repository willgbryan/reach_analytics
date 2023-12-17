import React from "react";

function MarketingPage() {
    const [animationStage, setAnimationStage] = React.useState(0);
    const [text, setText] = React.useState('Reach');
    const [buttonVisible, setButtonVisible] = React.useState(true);
    const triggerAnimation = () => {
      setButtonVisible(false);
      setText('reach');
      setTimeout(() => {
        const prefix = "Let's extend your ";
        let typedText = '';
        for (let i = 0; i < prefix.length; i++) {
          setTimeout(() => {
            typedText += prefix[i];
            setText(typedText + 'reach');
            if (i === prefix.length - 1) {
              setTimeout(() => setAnimationStage(2), 1000);
            }
          }, i * 100);
        }
        setAnimationStage(1);
      }, 500);
    };
    const resetAnimation = () => {
      setText('Reach');
      setButtonVisible(true);
      setAnimationStage(0);
    };
    React.useEffect(() => {
      let timeout;
      if (animationStage === 2) {
        timeout = setTimeout(() => {
          setAnimationStage(3);
        }, 1500);
      }
      return () => clearTimeout(timeout);
    }, [animationStage]);
    const textClass = `transition-opacity duration-1000 ease-in-out text-5xl font-bold cursor-pointer text-white ${
      animationStage === 2 ? 'opacity-0' : ''
    }`;
    return (
      <div className="relative w-full h-screen flex items-center justify-center" style={{ background: 'linear-gradient(to top, #ff7300, #74e8f9 30%, black 70%)' }}>
        {animationStage < 3 && (
          <div
            className={textClass}
            style={{
              transitionDelay: animationStage === 2 ? '0s' : '1s',
            }}
          >
            <span>{text}</span>
          </div>
        )}
        {animationStage === 3 && (
          <div className="absolute top-5 left-5">
            <button
              className="text-white"
              onClick={resetAnimation}
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="currentColor" className="bi bi-house-fill w-6 h-6" viewBox="0 0 16 16">
                <path fillRule="evenodd" d="m8 3.293l-6 6V15h4v-4h4v4h4v-5.707l-6-6z"/>
                <path fillRule="evenodd" d="M7.293 1.293a1 1 0 0 1 1.414 0l7 7-1 1-7-7-7 7-1-1 7-7z"/>
              </svg>
            </button>
          </div>
        )}
        {buttonVisible && (
          <button
            className="transition-opacity duration-500 ease-in-out absolute text-white"
            style={{ bottom: '20%', right: '10%', opacity: buttonVisible ? 1 : 0 }}
            onClick={triggerAnimation}
          >
            We’ll all make it sooner or later, better get going -{">"}
          </button>
        )}
      </div>
    );
  }

  export default MarketingPage;