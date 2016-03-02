//http://forum.arduino.cc/index.php?topic=102193.0




void setup()
{
  Serial.begin(9600);
}

void loop()
{

  //Pots are connected in order:
  // A0, A1, A2, A3, A4, A5
  // 6,  1,  2,  4,  5,  3
  //So read in order:
  // A1, A2, A5, A3, A4, A0
  Serial.print(analogRead(A1), DEC);
  Serial.print(';');
  delay(10);
  Serial.print(analogRead(A2), DEC);
  delay(10);
  Serial.print(';');
  Serial.print(analogRead(A5), DEC);
  delay(10);
  Serial.print(';');
  Serial.print(analogRead(A3), DEC);
  delay(10);
  Serial.print(';');
  Serial.print(analogRead(A4), DEC);
  delay(10);
  Serial.print(';');
  Serial.print(analogRead(A0), DEC);

  
  Serial.println();  // start a new line every 6 readings
  delay(20000);
}



