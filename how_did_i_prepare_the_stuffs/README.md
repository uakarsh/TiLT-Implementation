### A big note here about the preparation of the stuffs. 

* The notebooks from 1 to 3.2 used generative approach to solve the problem. This was part from the idea that T5 transformers are based on the generative approach.

* However, in the conclusion section of the TiLT transformer, the authors have mentioned that they have used the extractive approach, which means predicting the logits, and I have missed that part right now. I have added the code for preparing the FUNSD abstractive dataset, as well as the same would be followed for the CORD dataset.

* Also, I have the code for DocVQA (for extractive tasks, which includes predicting the start and the end logits of the answer from the context) ready, and I would also add it soon

* It would take me a while, to prepare the modeling approach for abstractive approach (as when I was going to finish the generative approach, I visited the paper and saw that the authors have used the extractive approach). 

* The idea was, actually confusing, as I was also ready for using the abstractive approach, but when I saw the T5's approach, I guess it hit me, and made me do the generative approach. Although, all the code are ready, I guess I would take a stop, and visit the abstractive approach for now. Let's see how this goes.

* By the way, if time permits, I would soon add the code for FUNSD, CORD as well as DocVQA, since I have worked on them, and have the idea to finetune the model on the same.