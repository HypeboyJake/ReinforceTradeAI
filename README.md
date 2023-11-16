# Easy-to-Use ReinforceTradeAI

Hello!
안녕하세요!

I have implemented trading using reinforcement learning.
I made it as simple as possible to use.
It can be used with just a CSV file.
강화학습을 이용한 트레이딩을 구현해보았습니다.
최대한 사용방법이 간단하게 만들었습니다.
csv파일만 있으면 사용가능합니다.

Please adhere to the CSV file format.
0th is time, 3rd is closing price, the number of features doesn't matter!
csv파일형식을 지켜주셔야합니다
0번은 시간, 3번은 종가 피쳐수는 상관없습니다!

There are two main files and env files.
If the first letter is 's', it's for stocks, and 'c' for crypto.

Open the env file for your intended use and modify appropriate values such as initial capital and commission.
The stock default is $1000 initial capital, 0.0% commission, and a minimum order amount of 1 share.
The crypto default is $1000 initial capital, 0.014% commission, and a minimum order amount of $10.

Go into the agent file in the sharing folder and select the model and optimizer you want to use.
The default is the PPO model and Adam optimizer.

Open the main file for your intended use and set it up according to the annotations!
Then execute it, and you're done.

If it's within your capability, try modifying the reward function in the env file and use it.

1. 두개의 main, env 파일이 있습니다.
    첫글자가 s면 주식이고, c면 크립토입니다.
   
3. 사용하고자하는 목적의 env파일을 열고 초기자본금, 수수료등 적절한값을 수정해주세요
    주식 디폴트는 초기자본금 1000달러, 수수료 0.0% 최소주문금액 1주
    크립토 디폴트는 초기자본금 1000달러, 수수료 0.014% 최소주문금액 10달러
   
5. sharing폴더의 agent파일로들어가 사용할 모델과 옵티마이저를 선택합니다.
    디폴트는 ppo모델과 adam옵티마이저
   
7. 사용하고자하는 목적의 main파일을 열고 각주에맞게 설정해주세요!
   그리고 실행하면 완료입니다.

![ex crypto](ex_crypto_img.png)
![ex stock](ex_stock_img.png)

#**<Caution>**
#**This project is an open-source sharing of a personally developed project,**
#**and all responsibilities for arising issues lie with the individual.**
#**<주의>**
#**본 프로젝트는 개인적으로 개발한 프로젝트를 오픈소스로 공유한 것으로**
#**발생하는 문제에 대한 모든 책임은 본인에게 있습니다.**
