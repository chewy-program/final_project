# Steam Price Predictor
The App is designed to predict the current cost of a Steam application using data gathered from the Steam API 

The app is deployed at this address:  

https://whispering-wave-32368.herokuapp.com/predict

The entering stage will look like the below: 

![image](https://user-images.githubusercontent.com/83536188/151108613-8616efce-4c49-49cd-96f4-91bf8e6294cf.png)

where 0 and 1 are used in place of True and False for Multiplayer and DLC 

On predict, the prediction will appear under it: 

![image](https://user-images.githubusercontent.com/83536188/151108734-bcbb0d1f-54ac-471f-b12d-54dc01d44587.png)
 
The price and initial price can be found here: 

![image](https://user-images.githubusercontent.com/83536188/151108406-647df154-d4fb-40d9-832a-a71aa76028a8.png)

![image](https://user-images.githubusercontent.com/83536188/151108277-814b0db9-2f44-42cf-9eca-3681e7c2ab3f.png)

the price as it relates to new columns created can be found here: 

![image](https://user-images.githubusercontent.com/83536188/151108371-1037d57a-2978-4d00-858c-4121cf68c75f.png)

![image](https://user-images.githubusercontent.com/83536188/151108387-635764b0-d22c-44bd-9a6e-4f5877c22f48.png)

![image](https://user-images.githubusercontent.com/83536188/151108399-ab2c13a0-9e9f-42d9-a03d-b20441a490ee.png)

The correlation of all columns can be found here: 

![image](https://user-images.githubusercontent.com/83536188/151108348-00c89bc6-a99c-4890-bbf1-c870caec5b73.png)

![image](https://user-images.githubusercontent.com/83536188/151108360-f38a1555-e0fb-4933-b534-d9fd6d06e7be.png)

and the residual plots of all the different models can be found below: 

![image](https://user-images.githubusercontent.com/83536188/151108433-de60f46b-fb0b-41cc-bd57-0143bcc5c7fc.png)

![image](https://user-images.githubusercontent.com/83536188/151108460-9349ac5d-d499-4e43-b37b-3cfa4a7c11e6.png)

![image](https://user-images.githubusercontent.com/83536188/151108475-63ac682f-aa02-4cc2-ab47-3a9576b224a3.png)


Overall I can conclude that the model is accurate to $20, then becomes less accurate as the value increases. 
