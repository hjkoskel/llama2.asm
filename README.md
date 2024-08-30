# llama2.asm

<p align="center">
  <img src="llamapic.png" width="300" height="300">
</p>

Goal of this project was to learn modern 64bit x86 assembly. I got previous personal experience on 16bit x86 assembly on dos. 
Re-implementing Karpathy's llama.c in assembly looked nice way to learn to solve different kind of data processing problems in asm. String handling and numerical operations.

Project scope is limited to single core.

Code is written completely in intel syntax assembly on nasm. Or I could say that this is more like debugged in gdb than written in asm.  Only 64bit linux is supported on purpose.


## Limitations and todos

- Single core
- kv_dim<dim , where  kv_dim = (dim * n_kv_heads) / n_heads;
    so case where n_kv_heads<n_head have to be implemented
- Implement following c-library functions
    - expf
    - powf
    - calloc
    - sinf
    - cosf
    - printf 
- Clean up comments in finnish
- clean up some commands, movs etc added for debug
- sampling is done without randomness, argmax sampling
    - Ok when developing asm code and making sure that end result is re-producable
- Performance is bad, this is continous journey
- Some other architehtures like ARM64bit

## Compiling and installing

For development following 
~~~ sh
nasm -g -l run.lst -felf64 run.asm && gcc -static -Wall run.o -g -lm && ./a.out
~~~

~~~ sh
nasm -felf64 run.asm && gcc -static run.o -lm && ./a.out
~~~

Program requires files named **model.bin** and **tokenizer.bin** placed on same dir as executable
Those are model files. There are many possible places to download those

One place to load models is on huggingface hub [tinyllamas](https://huggingface.co/karpathy/tinyllamas)


## Use examples
When program starts it will wait input from stdin. Input is used as prompt.

Press return and give empty prompt. Tinystories will print this
~~~txt
<s>
 Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to knock the ball down.
Lily found a stick and tried to hit the ball. But the stick was too short. She tried again and again, but she couldn't reach it. She felt sad.
Suddenly, a kind man came by and saw Lily. He asked her what was wrong. Lily told him about the ball. The man smiled and said, "I have a useful idea!" He took out a long stick and used it to knock the ball down. Lily was so happy! She thanked the man and they played together in the sunshine.
~~~


## Structure of code

It is easy to follow code by labels. Program starts at *main:*. Code will
- allocates loads model by *loadAI*
- read line from stdin to *promptext*
- *encode* string to token array (32bit per value), to *prompttokens*, tokencount stored to *promptokencount*
- If there is prompt, feed prompt tokens and position counter to *doForward* function so neural net works on *feedpromptloop:* loop
- run prediction loop until there is token with value 1 or max limit is reached
    - print previously aquired token. (from prompt or prediction process)
    - feed previous token and position to *doForward*, get logits
    - Use simple function *sample_argmax* for choosing token. Instead of this there could be selection function that uses random values and temperature. Get token from here 
    - free memory TODO

### Allocate by *loadAI*
Code loads checks size of model file with fseek. Based on that memory is allocated. File is loaded to two places. To *modelHeader* struct and to memory area *weights*

Some much used dimensions based on header are pre-calculated like *dim* * *hidden_dim*

Also some lookup tables for accelating interference
- *fcrfciArr*
- *freqArr*
- Size requirement of *statemem* is calculated by header and single memory are is callocated
- Pointers to *weights* memory are assigned. Those pointer variables have *wei_* prefix
- Memory is callocated for tokenizer
    - *tokenTextsAddr*  fixed size of 32 bytes per token (easy to access by multiplying token number 32 and adding tokenTextsAddr)
    - *tokenScoresAddr* token scores, each 4bytes (float32)
    - *tokenLenAddr* token length lookup, one byte per token
    - *tokenListsByLen* special 
    - *oneByteTokenLookup* is fixed size of 256, and it is does not require callocate
- Token file is read *vocabReadLoop:* progress token by token
    - text,scores and lengths
    - store index counter to *oneByteTokenLookup* (position pointed by byte) if length of token text is one byte.
- When token file is closed, special *tokenListsByLen* is built

Idea is to first list pointers... each points list of tokens with same length. Each of list is termintated with 0xFFFF.
This allows faster byte pair pairing when len(A)+len(B)=len(C).. Finding C is easier from list of correct length tokens than scanning thru whole 32000 long list of tokens.

### Tokenization by *encode:*

Tokenization is process where string is encoded to array of integers. Each integer is token. Token is few 

There are following phases
1. Replace each character on string with one byte token  (*encode:*)
2. Lump two tokens next to each others to one token (*mergeTokensStage:*)
    - new token text must be exactly match to concatted of two tokens to be merged (*call tokenPairLookup*)
    - token score for must be the highest among all pairs inside token array (*candidatewins:*)
3. Start from step 2 until there are noting to lump together

### Run network by *doForward:*

Big differences in between C implementation and this are

- Pointer address calculations with multiplications are replaced with add operations on each "step".
- fci fcr  (that freq thing) are calculated on each position so those sinf, cosf and powf are not needed to calculate on each layer