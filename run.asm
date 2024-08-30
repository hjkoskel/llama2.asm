;nasm -g -l assy.lst -felf64 assy.asm && gcc -static -Wall assy.o -g -lm && ./a.out
; Tokenization and decoding function in asm
bits 64
default rel
section .data
    align   16   
    ones dd 1.0, 1.0, 1.0, 1.0
    epsilon dd 0.00001

    filenameModel db "model.bin",0
    filenameTokenizer db "tokenizer.bin",0
    filehandleTokenizer db 8

    promptextlen equ 1024

    startTime_sec dq 1 ; 8 bytes for seconds (64-bit)
    startTime_nsec dq 1 ; 8 bytes for nanoseconds (64-bit)

    fmtTextTokensPerSec db 10,10,"Achieved tok/s: %d",10,0

section .bss
    align 16
    weights resq 1; model weights
    statemem resq 1 ;important to have 16 alligned
    
    ;puskuri resb 8  ;8tavua, 64 bittiä menee rekkariin
    ;numero resb 4 ; 32 tavua unsigned

    promptext resb 1024  ;TODO how long?
    promptokencount resq 1; ;64bit value
    prompttokens resw 1024 ;1024 x 16bit?

    struc modelHeaderStruct
        .dim: resb 4 ;transformer dimension
        .hiddenDim: resb 4 ; for ffn layers
        .n_layers: resb 4 ; // number of layers
        .n_heads:  resb 4  ; // number of query heads
        .n_kv_heads: resb 4 ; // number of key/value heads (can be < query heads because of multiquery)
        .vocabSize: resb 4 ;// vocabulary size, usually 256 (byte-level)
        .seq_len: resb 4 ;; // max sequence length    
    endstruc

    modelHeader         resb   modelHeaderStruct_size ;actually is config 7*4 is not 16 alligned
    head_size resq 1 ;Precalc dim/n_heads
    kv_dim resq 1 ;precalc  (dim * n_kv_heads) / n_heads;
    kStep resq 1 ; // precalc (c->dim*c->n_kv_heads) / (c->n_heads*c->n_heads)
    sqrt_head_size resq 1 ;as 32bit float
;--- size lookups----
    size_voc_dim resq 1  ;= vocab_size * dim; //Assemblyssä *4
    size_layers_dim resq 1  ;n_layers * dim;
    size_layers_dim_dim resq 1  ; n_layers * dim *  dim;
    size_layers_dim_hiddenDim resq 1  ; n_layers * dim * hidden_dim;
    size_layers_dim_nkvheads_headSize resq 1  ; n_layers * dim * n_kv_heads * head_size;
    size_layers_dim_heads_headSize resq 1  ;n_layers * dim * n_heads * head_size;

    size_dim_dim resq 1 ;dim*dim
    size_dim_hiddenDim resq 1 ;dim*hidden_dim
    size_dim_kvDim resq 1 ;dim*kvDim
    size_seqlen_kvDim resq 1; seq_len * kv_dim

;---- lookups precalced
    freqArr resq 1 ; lookup dim/2 for fcr fci cal.  1.0f / powf(10000.0f, (i % (head_size)) / (float)(head_size))
    fcrfciArr resq 1  ; Goes in pairs  fcr,fci, fcr,fci, fcr,fci...  UPDATED ON EACH POS!
 
    modelfilelen    resq 1 ;Just number how
    ;Nope 16byte allignment use separate! modelcontent    resq 1 ; file is stored at here
    

    tokenTextsAddr      resq 1   ; tokens text table starts he
    tokenScoresAddr     resq 1  ; 32bit floats, 4bytes
    tokenLenAddr resq 1 ; 
    tmpLen      resd 4 ;read from file... FYI todo test?

    tokenListsByLen resq 1 ; 32 64bit pointers to memory 0 and 1 are not used 0xFFFF termination

    oneByteTokenLookup resd 256  ; Faster to make initial byte->token conversion


    modelFileHandle resq 1
    tokenizerFileHandle resq 1

    workBuffer resq 1 ; pointer to 32vi
    workBufferLen resq 1 ; 64bit number how many tokens

    ;tokenizer recordin
    bpeBestFirstPos resq 1 ;address in memory best candicate
    bpeBestToken resb 2; 16bit value
    ;XMM1 bpeBestScore resb 4; 32bit float, for best score of token

    firstToken resq 1; //64bit easy for reg oper
    secondToken resq 1;

;-----------State memory pointers. All these point to float array ------------
    ;// current wave of activations
    x resq 1; // activation at current time stamp (dim,)
    xb resq 1; // same, but inside a residual branch (dim,)
    xb2 resq 1; // an additional buffer just for convenience (dim,)
    hb resq 1; // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2 resq 1; // buffer for hidden dimension in the ffn (hidden_dim,)
    q resq 1; // query (dim,)
    k resq 1; // key (dim,)
    v resq 1; // value (dim,)
    att resq 1; // buffer for scores/attention values (n_heads, seq_len)
    logits resq 1; // output logits
    ;// kv cache
    key_cache resq 1;   // (layer, seq_len, dim)
    value_cache resq 1; // (layer, seq_len, dim)
;--------------- Weights pointers ---------------
    wei_token_embedding_table resq 1;   // (vocab_size, dim)
    ;weights for rmsnorms
    wei_rms_att_weight resq 1; // (layer, dim) rmsnorm weights
    wei_rms_ffn_weight resq 1; // (layer, dim)
    ;weights for matmuls. note dim == n_heads * head_size
    wei_wq resq 1; // (layer, dim, n_heads * head_size)
    wei_wk resq 1; // (layer, dim, n_kv_heads * head_size)
    wei_wv resq 1; // (layer, dim, n_kv_heads * head_size)
    wei_wo resq 1; // (layer, n_heads * head_size, dim)
    ;weights for ffn
    wei_w1 resq 1; // (layer, hidden_dim, dim)
    wei_w2 resq 1; // (layer, dim, hidden_dim)
    wei_w3 resq 1; // (layer, hidden_dim, dim)
    ;final rmsnorm
    wei_rms_final_weight resq 1; // (dim,)
    ;(optional) classifier weights for the logits, on the last layer
    wei_wcls resq 1;

;-------- pointers on forward method re-set on each forward, add on each layer -------
    point_rmsnorm resq 1 ;reset wei_rms_att_weight   add dim 
    point_wq resq 1  ;reset wei_wq  Add dim*dim
    point_wk resq 1  ;reset wei_wk  Add dim*kv_dim
    point_wv resq 1  ;reset wei_wk  Add dim*kv_dim
    point_wo resq 1  ;reset wei_wo  Add dim*kv_dim
    
    ;float *point_ffn_weight=wei_rms_ffn_weight;
    point_ffn_weight resq 1 ;reset wei_rms_ffn_weight  add dim;

    point_w1 resq 1  ;reset wei_w1, Add dim*hidden_dim
    point_w2 resq 1  ;reset wei_w2, Add dim*hidden_dim
    point_w3 resq 1  ;reset wei_w3, Add dim*hidden_dim

;----- pointers, reset on each layer iteration,    add on each head loop
    point_q resq 1            ;reset q ,               add head_size
    point_att resq 1          ;reset att,              add seq_len 
    point_k_layer_head resq 1 ;reset key_cache + loff, add kStep
    point_xb resq 1           ;reset xb,               add head_size

    point_k_layer_head_pos resq 1

    loff resq 1; helper variable
    var_pos resq 1; pos parameter..
    point_layer_value resq 1

    clockvalue resq 1
section .text


global main

;TODO implement these functions internally. Just syscalls!
extern expf
extern powf
extern calloc
extern sinf
extern cosf
extern printf  ;used only when printing tok/s


;***********************
; loadAI
; Allocates, loads and initializes all
;***********************
loadAI:
    push RBP
    mov RBP,RSP
    sub RSP,128
    push RBX
    push R12
    push R13


    ;-- Open Model file ---
    mov RAX,2
    mov RDI, filenameModel
    mov rsi,0 ; flags, readonly
    mov rdx,0644o ;mode
    syscall ; OPEN model
;---  fseek(file, 0, SEEK_END);
    cmp rax, 0 ; check for success
    jl fileOpenErrExit
    mov [modelFileHandle],rax
    mov rdi, rax ;fd to parametr
    mov rax, 8                     ; sys_lseek syscall number (8)
    mov rsi, 0                     ; Offset (0)
    mov rdx, 2                     ; Whence: SEEK_END (2)
    syscall
    cmp rax, 0 ; check for success
    jl fileOpenErrExit
    ;SIZE OF FILE IS ON RAX
    mov [modelfilelen],rax

    ;rewind.... to start
    mov rdi, [modelFileHandle]
    mov rax, 8                     ; sys_lseek syscall number (8)
    mov rsi, 0                     ; Offset (0)
    mov rdx, 0                     ; Whence: from start
    syscall
    cmp rax, 0 ; check for success
    jl fileOpenErrExit
    ;lets calloc memory and fill up
    mov edi,[modelfilelen]
    sub edi,28
    mov rsi,1 ;modelfilelen*1byte
    call calloc
    mov [weights],rax

    ;--- Read header ---
    mov rdi,[modelFileHandle]
    mov rax, 0 ;SYS_READ
    mov rsi, modelHeader
    mov rdx, 28
    syscall ; read model header
    cmp rax, 0 ; check for success
    jl fileOpenErrExit
    ;--- Read weights ---
    mov rdi,[modelFileHandle]
    mov rax, 0 ;SYS_READ
    mov rsi, [weights]
    mov rdx, [modelfilelen]
    sub rdx,28
    syscall ; read model header
    cmp rax, 0 ; check for success
    jl fileOpenErrExit
    ;--- close file ---
    mov rax,3 ; CLOSE
    mov rdi,[modelFileHandle]
    syscall ; close header file
    cmp rax, 0 ; check for success
    jl fileOpenErrExit

    ;---Precalc some values from conf
    xor rdx,rdx
    mov eax,[modelHeader+modelHeaderStruct.dim]
    div dword [modelHeader+modelHeaderStruct.n_heads]
    mov [head_size],rax  ;head_size resq 1 ;Precalc dim/n_heads
    
    mov eax,[modelHeader+modelHeaderStruct.dim]
    mul dword [modelHeader+modelHeaderStruct.n_kv_heads]
    div dword [modelHeader+modelHeaderStruct.n_heads]
    mov [kv_dim],rax

    mov eax,[modelHeader+modelHeaderStruct.dim]
    mul dword [modelHeader+modelHeaderStruct.n_kv_heads]
    div dword [modelHeader+modelHeaderStruct.n_heads]
    div dword [modelHeader+modelHeaderStruct.n_heads]
    mov [kStep],rax
    
    cvtsi2ss xmm0,rax
    sqrtss xmm0,xmm0
    movss [sqrt_head_size],xmm0
    ;kStep resq 1 ; // precalc (c->dim*c->n_kv_heads) / (c->n_heads*c->n_heads)

    ;some size calculations
    mov eax,[modelHeader+modelHeaderStruct.vocabSize]
    mul dword [modelHeader+modelHeaderStruct.dim]
    shl rax,2 ; float 4bytes
    mov [size_voc_dim],rax  ;= vocab_size * dim; //Assemblyssä *4

    mov eax,[modelHeader+modelHeaderStruct.n_layers]
    mul dword [modelHeader+modelHeaderStruct.dim]
    shl rax,2 ; float 4bytes
    mov [size_layers_dim],rax  ;n_layers * dim*4;

    mul dword [modelHeader+modelHeaderStruct.dim]
    mov [size_layers_dim_dim],rax  ; n_layers * dim *  dim;

    mov eax,[modelHeader+modelHeaderStruct.n_layers]
    mul dword [modelHeader+modelHeaderStruct.dim]
    mul dword [modelHeader+modelHeaderStruct.hiddenDim]
    shl rax,2 ; float 4bytes
    mov [size_layers_dim_hiddenDim],rax  ; n_layers * dim * hidden_dim;

    mov eax,[modelHeader+modelHeaderStruct.n_layers]
    mul dword [modelHeader+modelHeaderStruct.dim]
    mul dword [modelHeader+modelHeaderStruct.n_kv_heads]
    mul dword [head_size]
    shl rax,2 ; float 4bytes
    mov [size_layers_dim_nkvheads_headSize],rax  ; n_layers * dim * n_kv_heads * head_size;

    mov eax,[modelHeader+modelHeaderStruct.n_layers]
    mul dword [modelHeader+modelHeaderStruct.dim]
    mul dword [modelHeader+modelHeaderStruct.n_heads]
    mul dword [head_size]
    shl rax,2 ; float 4bytes
    mov [size_layers_dim_heads_headSize],rax  ;n_layers * dim * n_heads * head_size;

    mov eax,[modelHeader+modelHeaderStruct.dim]
    mul dword [modelHeader+modelHeaderStruct.dim]
    shl rax,2 ; float 4bytes
    mov [size_dim_dim],rax

    mov eax,[modelHeader+modelHeaderStruct.dim]
    mul dword [modelHeader+modelHeaderStruct.hiddenDim]
    shl rax,2 ; float 4bytes
    mov [size_dim_hiddenDim],rax
    
    mov eax,[modelHeader+modelHeaderStruct.dim]
    mul dword [kv_dim]
    shl rax,2 ; float 4bytes
    mov [size_dim_kvDim],rax ;dim*kvDim

    mov eax,[modelHeader+modelHeaderStruct.seq_len]
    mul dword [kv_dim]
    shl rax,2 ; float 4bytes
    mov [size_seqlen_kvDim],rax ;seqlen*kvDim


    ;--- calclookups ---
    mov edi, dword [modelHeader+modelHeaderStruct.dim]
    mov rsi,4 ; floats
    call calloc
    mov [fcrfciArr],rax  ;JUST RESERVE HERE, UPDATE WHEN POS UPDATES

    ;---allocate frequency lookup table ---
    mov edi, dword [modelHeader+modelHeaderStruct.dim]
    mov rsi,2 ; floats/2
    call calloc
    mov [freqArr],rax  ;JUST RESERVE HERE, UPDATE WHEN POS UPDATES

    mov ecx, dword [modelHeader+modelHeaderStruct.dim]
    ;TODO CALC 1.0f / powf(10000.0f, (i % (head_size)) / (float)(head_size));

    mov edi,dword [modelHeader+modelHeaderStruct.dim]
    shl rdi,1 ;  4/2 *float because dim/2 is length
    add rdi,[freqArr]

    xorps xmm0,xmm0
    xorps xmm1,xmm1 ;easy debug?
fcrfciprecalcLoop:
    dec rcx
    dec rcx

    sub rdi,4 ;move by one float

    mov rax,rcx          ; if rax=286, rbx=48  -> edx=46    286%48= 46
    mov ebx,[head_size]
    xor rdx,rdx
    div rbx
    ;dx= i%headSize
    cvtsi2ss xmm1,edx  ;v4_float = {46, 0, 0, 0}
    cvtsi2ss xmm0, dword [head_size] ;v4_float = {48, 0, 0, 0}
    divss xmm1,xmm0 ; xmm1=i % (head_size)) / (float)(head_size)    xmm0= v4_float = {0.958333313, 0, 0, 0}

    mov edx,10000
    cvtsi2ss xmm0,edx
    push rdi
    push rcx  ;xmm0:v4_float = {10000, 0, 0, 0}  xmm1: v4_float = {0.958333313, 0, 0, 0}
    call powf
    pop rcx ;v4_float = {6812.91943, 5.83165503, 0, 0}    6812.919414676924
    pop rdi ;pos
    mov edx,1
    cvtsi2ss xmm1,edx
    divss xmm1,xmm0
    ;rcpss xmm0,xmm0 ; now xmm0=freq=1/powf   OK 286 -> 0.000146776438    
    movss [rdi],xmm1  ; v4_float = {0.000146776438, 5.83165503, 0, 0}
    dec rcx  
    inc rcx
    inc rcx
    loop fcrfciprecalcLoop


    ;----- Allocate state --------------------

    xor rdi,rdi ;count goes here How many floats

    mov eax,[modelHeader+modelHeaderStruct.dim]
    shl rax,2 ; *4
    add rdi,rax  ;rdi=1152, rax=1152 

    mov eax,[modelHeader+modelHeaderStruct.hiddenDim] ;768
    shl rax,1 ; *2
    add rdi,rax ;rdi=2688

    xor rdx,rdx ;need to zero?
    mov eax,[modelHeader+modelHeaderStruct.n_layers]
    mul dword [modelHeader+modelHeaderStruct.seq_len]
    mul dword [modelHeader+modelHeaderStruct.dim]
    mul dword [modelHeader+modelHeaderStruct.n_kv_heads]
    shl rax,1 ; *2
    div dword [modelHeader+modelHeaderStruct.n_heads]
    add rdi,rax

    mov eax,[modelHeader+modelHeaderStruct.n_heads]
    mul dword [modelHeader+modelHeaderStruct.seq_len]
    add rdi,rax

    add edi,[modelHeader+modelHeaderStruct.vocabSize]

    mov rsi,4 ;sizeof(float)
    call calloc
    mov [statemem],rax ;main pointer
    ;--- Assign pointers pointing to state memory ---
    mov rcx,rax ; use now this as pointer,  AX is for sizes
    mov [x],rcx; activation at current time stamp (dim,)
    xor rdx,rdx ; Call caused something?

    mov eax,dword [modelHeader+modelHeaderStruct.dim]
    shl rax,2 ;float size
    add rcx,rax
    mov [xb],rcx ;same, but inside a residual branch (dim,)

    add rcx,rax ; 4*dim
    mov [xb2],rcx

    add rcx,rax ; 4*dim
    mov [hb],rcx

    mov eax,dword [modelHeader+modelHeaderStruct.hiddenDim]
    shl rax,2;float size
    add rcx,rax
    mov [hb2],rcx
   
    add rcx,rax  ;is hiddedDim*2
    mov [q],rcx

    ; USED AS CHANCING POINTER   k resq 1; // key (dim,)
    ; USED AS CHANCING POINTER  v resq 1; // value (dim,)
    ; USED AS CHANCING POINTER  att resq 1; // buffer for scores/attention values (n_heads, seq_len)
    ; USED AS CHANCING POINTER   logits resq 1; // output logits
    ;// kv cache

    mov eax,dword [modelHeader+modelHeaderStruct.dim]
    shl rax,2;float size
    add rcx,rax
    mov [key_cache],rcx

    mov eax,dword [modelHeader+modelHeaderStruct.n_layers]
    mul dword [modelHeader+modelHeaderStruct.seq_len]
    mul dword [kv_dim]
    shl rax,2;float size
    add rcx,rax
    mov [value_cache],rcx
    add rcx,rax  ;n+=c->n_layers * c->seq_len * (kv_dim);
    mov [att],rcx

    mov eax,dword [modelHeader+modelHeaderStruct.n_heads]
    mul dword [modelHeader+modelHeaderStruct.seq_len]
    shl rax,2;float size
    add rcx,rax
    mov [logits],rcx

    ;-- assignWeightPointers --

    mov rcx,[weights]
    mov [wei_token_embedding_table],rcx
    mov [wei_wcls],rcx  ;wei_wcls = shared_weights ? wei_token_embedding_table : ptr;

    ;weights for rmsnorms
    add rcx,[size_voc_dim]
    mov [wei_rms_att_weight],rcx

    add rcx,[size_layers_dim]
    mov [wei_wq],rcx

    add rcx,[size_layers_dim_dim]
    mov [wei_wk],rcx

    add rcx,[size_layers_dim_nkvheads_headSize]
    mov [wei_wv],rcx

    add rcx,[size_layers_dim_nkvheads_headSize]
    mov [wei_wo],rcx

    add rcx,[size_layers_dim_heads_headSize]
    mov [wei_rms_ffn_weight],rcx

    add rcx,[size_layers_dim]
    mov [wei_w1],rcx

    add rcx,[size_layers_dim_hiddenDim]
    mov [wei_w2],rcx

    add rcx,[size_layers_dim_hiddenDim]
    mov [wei_w3],rcx

    add rcx,[size_layers_dim_hiddenDim]
    mov [wei_rms_final_weight],rcx
    
    ;-----------------------
    ;--- loadTokenizer -----
    ;-----------------------
    ; 32 bytes per entry
    ;---ALLOCATE TEXT----
    mov edi, dword [modelHeader+modelHeaderStruct.vocabSize]
    mov rsi,32 ; ONKO LIIKAA 32 tarvii
    call calloc
    mov [tokenTextsAddr],rax
    ;---ALLOCATE SCORES -----
    mov edi, dword [modelHeader+modelHeaderStruct.vocabSize]
    mov rsi,4
    call calloc
    mov [tokenScoresAddr],rax
    ;---ALLOCATE LENGTH LOOKUP---
    mov edi, dword [modelHeader+modelHeaderStruct.vocabSize]; 32bit int but needed 8BIT 0-31 chars
    mov rsi,1; //One byte is enough!
    call calloc
    mov [tokenLenAddr],rax
    ;---ALLOCATE SPACE FOR listing tokenID:s by length. allocate number of tokens. Assume at least 32 of lenght of 1 tokens
    mov edi, dword [modelHeader+modelHeaderStruct.vocabSize];; 32bit size, 16 bit needed
    add edi,256 ; For alloc table 64bit pointers  (easier than index?)
    mov rsi,2 ; index 16bit is enough for 32000
    call calloc
    mov [tokenListsByLen],rax

    ;--- open tokenizer file ----
;opentokenizerfile:
    mov RAX,2 ; open syscall function number
    mov RDI, filenameTokenizer
    mov rsi,0 ; flags, readonly
    mov rdx,0644o ;mode
    syscall ; open tokenizer
    mov [tokenizerFileHandle],rax
    cmp rax, 0 ; check for success
    jl fileOpenErrExit
    
    mov rdi,rax ; File handle
    xor rax,rax
    mov rsi, tmpLen ;Now got max_token_length, not needed
    mov rdx,4
    syscall ;read
    mov ecx, dword [modelHeader+modelHeaderStruct.vocabSize];; 32bit using rcx
    ;mov rcx,32000
    ;Each entry have float of scores
    xor rbx,rbx
vocabReadLoop: ;Each entry have length of string (int)
    push rbx
    push rcx ; token number
   
    ;------ Read scores -----------
    ;calculate position 32bit score on array
    mov eax,dword [modelHeader+modelHeaderStruct.vocabSize];
    lea rsi,[eax]
    shl rsi,2 ;4bytes size
    add rsi, [tokenScoresAddr]  ;Nyt  4*vocasize+tokenScoresAddr arvo 
    shl rcx,2 ; counter, for 4bytes per entry
    sub rsi,rcx ;counter starts from up so subtract from end ; nyt 4*vocabsize+addr-counter*4
    mov rdi,[tokenizerFileHandle] ;file handle
    xor rax,rax
    mov rdx,4 ;read just 32bit float
    syscall ;Now f32 is placed in token scores array

    ;----Then get length of token string
    ;-This commented out version discards length
    mov rsi, tmpLen
    mov rdi,[tokenizerFileHandle] ;file handle
    xor rax,rax
    mov rdx,4 ;read just int32
    syscall
    ;just copy to correct index
    pop rcx
    push rcx

    mov esi,dword [modelHeader+modelHeaderStruct.vocabSize];
    add rsi, [tokenLenAddr] 
    sub rsi,rcx
    mov eax, dword [tmpLen];
    mov [rsi],al ;8bit is enough
    ;Now token is stored

    ;Calculate position on token string table and read directly there
    pop rcx
    push rcx

    mov esi,dword [modelHeader+modelHeaderStruct.vocabSize] 
    shl rsi,5 ;32bytes
    add rsi, [tokenTextsAddr]  ;Nyt  32*vocasize+addr  
    shl rcx,5
    sub rsi,rcx ;counter starts from up so subtract from end ; nyt 32*vocabsize+addr-counter*32
    mov rdi,[tokenizerFileHandle] ;file handle
    mov edx, dword [tmpLen];
    push rsi ; Store text position
    xor rax,rax
    syscall
    pop rsi ; restore text position
    pop rcx ; for loop.. pun intended
    pop rbx
    ;----- building lookup for one byte tokens -------
    mov edx, dword [tmpLen];
    cmp edx,1
    jne moreThan1 ;Check if this 
    ;now it is oneByteToken
    mov al, byte [rsi] ;Copy single char to al... I mean as RAX
    shl rax,1 ;2bytes as token number
    add rax,oneByteTokenLookup ; Now rax points to correct place for lookup
    ;do not overwrite
    xor r8,r8
    mov r8w,word[rax]
    cmp r8,0
    jne moreThan1 ;helps to use lower token values, same as in original
    mov word[rax], bx
moreThan1:
    inc rbx
    dec rcx
    jnz vocabReadLoop  ;loop, too far lets use dec and jnz

    ;-- closeTokenizer ----
    mov rax,3 ; CLOSE
    mov rdi,[tokenizerFileHandle] ;filedescriptor goes here
    syscall

    ;--- precalcSizeLookup ----
    mov rcx,30 ;rcx is token size searched
    mov r12,[tokenListsByLen] 
    mov r11,[tokenListsByLen] ;32 64bit pointers. R11 osoittaa missä mennään. Väliin laitetaan 0xFFFFFFF:ää  TÄYTTÖREKKARI
    add r11,256
    xor r13,r13
precalcSizeLookup_loopLen:
    push rcx
    ;Tulostauluun kirjoitetaan mistä täyttö alkaa
    mov rax,rcx
    shl rax,3
    add rax,[tokenListsByLen] ;pointer to arr  TAI r12:sta
    mov rbx,r11
    sub rbx,[tokenListsByLen]
    mov [rax],rbx  ;starting point store
    ;Prepare tokenLenAddr loop thru. Loop from end. RCX is token number
    mov r9,rcx ; COPY HERE, needed for compare.. C register is used for loop  R9 tells what is searched
    mov ecx, dword [modelHeader+modelHeaderStruct.vocabSize] ;loop thru whole vocab size array. TOKEN COUNTER
precalcSizeLookup_loopVocab:
    mov r10,rcx
    add r10,[tokenLenAddr] ; TOKEN LEN IS ONE BYTE, indexing is 16bit
    xor rax,rax
    mov al, byte [r10] ;8bit is enough
    cmp rax,r9 ; haetaan 
    jnz precalcSizeLookup_nomatch
    ;MATCHING Write to rdi and step forward. Just assign token number to list
    mov [r11],rcx
    add r11,2; //NEXT! 16bit
    inc r13
precalcSizeLookup_nomatch:
    loop precalcSizeLookup_loopVocab
    mov word [r11],0xFFFF
    add r11,2; //NEXT! 16bit size
    pop rcx
    loop precalcSizeLookup_loopLen

    pop R13
    pop R12
    pop RBX
    add RSP,128
    pop RBP
    ret

fileOpenErrExit:
    pop R13
    pop R12
    pop RBX
    add RSP,128
    pop RBP
    ret

;****************
;** catmatch ****
;****************
;rdi:firstString,rsi:secondString, rdx:targetstring
catmatch:
    mov al,byte[rdi]
    cmp al,0
    je catmatchLoopSecond ; ok continue with second string
    mov bl,byte[rdx]
    cmp al,bl
    jne catnotmatch
    inc rdi
    inc rdx
    jmp catmatch

catmatchLoopSecond:
    mov al,byte[rdx]
    cmp al,0
    je catmatchok 
    mov bl,byte[rsi]
    cmp al,bl
    jne catnotmatch
    inc rsi
    inc rdx
    jmp catmatchLoopSecond

catmatchok:
    mov rax,1
    ret

catnotmatch:
    mov rax,0
    ret

;**********************
;** tokenPairLookup ***
;**********************
;Get token number that matches with two joined tokens in text
;rdi firstTok, rsi: secondTok
tokenPairLookup:
    push RBP
    mov RBP,RSP
    sub RSP,128
    push RBX
    push R12
    push R13
    push R14
    push R15
    
    mov rdx,[tokenLenAddr]
    xor rax,rax
    mov al,byte [rdi+rdx]
    add al,byte [rsi+rdx]
    ;RAX have now length
havelen:
    mov rdx,[tokenListsByLen]
    mov rax,[rdx+rax*8] ;rax is address of first item wanted length  NYT RAX:ssa on offset ekaan kandidaattiin
    add rax,[tokenListsByLen]
    ;now transform firstok and secondtok to string pointers. Token values are not needed anymore
    mov rcx,[tokenTextsAddr]
    shl rdi,5
    add rdi,rcx ;Just tokennumber*32+textMemoryStartAddress
    shl rsi,5
    add rsi,rcx

    push rax
candidateLoop:
    pop rax

    xor rbx,rbx
    mov bx, word [rax] ;rbx have now 16bit token candidate
    cmp bx,0xFFFF
    je exitTokenPairLookup
    ;ok, now  rbx have token number

    add rax,2
    push rax
    push rbx
    push rdi
    push rsi
    mov rdx,[tokenTextsAddr]
    shl rbx,5 ;32 bytes per entry. 
    add rdx,rbx ;now this have address of candidate text (null term)
    call catmatch
    pop rsi
    pop rdi
    pop rbx
tes:
    cmp rax,0
    je candidateLoop ;not found
    pop rax
exitTokenPairLookup:
    mov rax,rbx ;matched token
    pop R15
    pop R14    
    pop R13
    pop R12
    pop RBX
    add RSP,128
    pop RBP
    ret

;***********************
;***** ENCODE ********** 
;***********************
;(RBX, RBP, RSP and from R12 to R15 must be preserved by the called function)
;void encode(char *text, int *tokens, int *n_tokens);
;rdi: *text , rsi: *tokens, rdx: *n_tokens
encode:
    push RBP
    mov RBP,RSP
    sub RSP,128
    push RBX
    push R12
    push R13
    push R14
    push R15
    ;** Preliminary coding... assingn one token per byte ****
    ;r10->rdi
    mov [workBuffer],rsi ;Store start of buffer given
    ;mov dword [rdx],1
    mov dword [rsi],1
    add rsi,4
    mov rcx,1
initialTextToTokenLoop:
    xor rax,rax
    mov al,byte [rdi] ;get byte from input
    inc rdi

    cmp rax,0
    jz initialTokenizationDone ; nollaan päättyy inputti
    shl rax,1; raxissa on characterin numero 0-255 kerro 2:lla niin saa offsetin
    mov rbx,oneByteTokenLookup
    add rbx,rax
    xor rax,rax
    mov ax,word [rbx] ;get token
    mov dword [rsi],eax ; store to output array
    add rsi,4
    inc rcx ;inc length counter
    jmp initialTextToTokenLoop
initialTokenizationDone:
    mov [workBufferLen],ecx ;store here

    ;*** Merge tokens, consecutive tokens  A and B. Get C strlen(A)+strlen(B)=strlen(C) from list
    ;*** A and B concatted must be C. If multiple hits, choose with highest scoremergeTokensStage:
    ;alustellaan muuttujat
mergeTokensStage: ;start merging, this is call until there is nothing to merge
    mov r8w,0xFFFF
    mov word [bpeBestToken],r8w ; Mark not found with best 0xFFFF... instead of -1 or -1e10 on float use this as marker
    mov rcx,[workBufferLen]  ;it is ok to start from end? Not same as in C implementation  but simpler in this case
    ;avoid neg...
    cmp rcx,2
    jle handlebest
    dec rcx
    dec rcx
workBufferLoop: ; loop from start
    mov r8,[workBuffer]
    xor rdi,rdi
    xor rsi,rsi
    mov edi, dword[r8+rcx*4]
    mov esi, dword[r8+rcx*4+4] ;work buffer is 32bit ints
    push rcx ;keep work buffer index somewhere 
    push rdx
    call tokenPairLookup  ; rdi firstTok, rsi: secondTok
    pop rdx
    pop rcx
    cmp rax,0xFFFF ;Check return token value
    jne pairMatchFound  ;
    ;not found candidate, continue looping
    loop workBufferLoop  ;id==-1  continue
    jmp handlebest ;protects code after loop
pairMatchFound:
    ;check score... if better update that
    mov rbx,rax ;copy here
    shl rbx,2 ; 4 bytes per entry float32 is score
    add rbx,[tokenScoresAddr]
    movss XMM0,[rbx] ;copy...
    mov r8w, word [bpeBestToken]; If best token is not set yet
    cmp r8w,0xFFFF
    je candidatewins  ;first hit
    comiss XMM1,XMM0   ;xmm0,xmm1 oli eka
    jb candidatewins ;JB candidatewin
    loop workBufferLoop ;was not good enough...continue
    jmp handlebest ;protects code after loop
candidatewins:
    movss XMM1,XMM0
    mov [bpeBestToken], word ax 
    mov [bpeBestFirstPos],rcx
    loop workBufferLoop

    ;---looping work buffer is ready, lets check 
handlebest: ;Check if 
    mov r8w, word[bpeBestToken]
    cmp r8w,0xFFFF
    je retEncode ;Not found best at all in this loop thru -> exit
    ;got best.. do merge
    mov rbx,[bpeBestFirstPos]
    shl rbx,2 ; size is 4 bytes
    mov r8,[workBuffer]
    add r8,rbx

    xor rax,rax
    mov ax,word[bpeBestToken]
    mov [r8],word ax ;Place to work buffer
    ;Lasketaan montako siirtoa tulevasta tähän..alkaen second tokenista tarvitaan. Word kerrallaan
    mov rcx,[workBufferLen]
    sub rcx,[bpeBestFirstPos]
shiftloop:
    add r8,4
    xor rax,rax
    mov ax,word[r8+4]
    mov word [r8],ax
    loop shiftloop
    mov rcx,[workBufferLen] ;shrunk by one token
    dec rcx
    mov [workBufferLen],rcx
    jmp mergeTokensStage ;Lets re-start


retEncode:
    mov r8,[workBufferLen]
    mov word [rdx],r8w
    pop R15
    pop R14    
    pop R13
    pop R12
    pop RBX
    add RSP,128
    pop RBP
    ret ;returns, done, not found matchs anymore in this loop thru workbuffer


;**********************************
;*********** VECADD ***************
;**********************************
;void vecadd(float *x,float *y,int n);
vecadd: ;; rdi:*x , rsi:*y, rdx: n
    mov rcx,rdx
    shr rcx,2
vecaddloop:
    movaps xmm0,[rdi]
    addps xmm0,[rsi]
    movaps [rdi],xmm0
    add rdi,16
    add rsi,16
    loop vecaddloop
    ret

;********************
;**** DOTPRODUCT ****
;********************
dotproduct: ;; rdi:*x , rsi:*y, rdx: n
    mov rcx,rdx
    shr rcx,2
    xorps xmm0,xmm0 ;set sum to 0,0,0,0
dotproductloop:
    movaps xmm1,[rdi]
    mulps xmm1,[rsi]
    addps xmm0,xmm1
    add rdi,16
    add rsi,16

    loop dotproductloop
    haddps xmm0,xmm0
    haddps xmm0,xmm0 ; now This have scalar sum of 4 iterations
    ret
;**********************************
;********** RMSNORM  **************
;**********************************
rmsnorm:  ; rdi:*o , rsi:*x, rdx:*weight, rcx:size   (RBX, RBP, RSP and from R12 to R15 must be preserved by the called function)
    mov r11,rcx ; kopio talteen!!
    shr rcx,2 ;4 entries per loop  HUOM! PITÄÄ OLLA 4:lla jaollinen!
    mov r10,rsi
    xorps xmm1,xmm1 ; sum goes here
rmsnormloop:
    movaps xmm0,[r10]
    mulps xmm0,xmm0 ; Nyt on toiseen potenssiin mutta hajallaan
    addps xmm1,xmm0 ;ja sitten hajallaan oleva summa
    add r10,16; stepataan muistissa eteenpäin
    loop rmsnormloop
rmsnormlooprdy:
    haddps xmm1,xmm1
    haddps xmm1,xmm1  ;JEE!!! tää on hieno
    ;NYT for ss= x[j]*x[j]  on jokaisessa solussa. Pidetään nyt kopio kaikissa 4:ssä?  TODO BENCHMARKKAA MITEN VAIKUTTAA
sslaskettu: ;1496.20996 TÄSMÄÄ!
    cvtsi2ss xmm0,r11   ;xmm0 on floattien määrä!
    shufps xmm0,xmm0,0 ; Nyt size on kaikissa 4:ssä. TODO ONKO???
    divps xmm1,xmm0 ;  summed squares jaetaan sizellä.. kaikissa soluissa sama luku
epsi:
    mov eax, dword [epsilon]
    movd xmm0, eax
    shufps xmm0,xmm0,0 
    addps xmm1,xmm0

    rsqrtps xmm1,xmm1 ;1/sqrt:llä käsitelty nyt

    mov rcx,r11
    shr rcx,2 ;4 entries per loop
    mov r10,rsi ; x vektoriin osoitus
    ;MUISTA ETTÄ xmm1 sisältää ss:n
rmsnormloop2:
    movaps xmm0,[r10] ;x here
    movaps xmm2,[rdx] ;weight here
    mulps xmm0,xmm1  ; now xmm3 is  x[]*ss[]
    mulps xmm0,xmm2
    movaps [rdi],xmm0 ;return output to memory
    ;step to next memory block and loop
    add r10,16 ; step x pointer
    add rdx,16 ; step weight pointer
    add rdi,16 ; step output pointer
    loop rmsnormloop2
    ret

;**************
;*** MATMUL ***
;**************
matmul: ; rdi:*xout , rsi:*x, rdx:*w, rcx:n, r8:d  (RBX, RBP, RSP and from R12 to R15 must be preserved by the called function)
    mov r9,rcx ; copy rcx here
    mov rcx,r8; outer runs d, stored in r8
matmulOuterLoop: ;Loop d times, one loop per result.  4x thing is in inner
    xorps xmm0, xmm0 ; val=0 TODO KÄYTÄ TÄTÄ SKALAARINA?

    mov r8,rcx ;use reg not stack :D
    mov rcx,r9 ;init from stored reg
    mov r10,rsi ;start x[r10] iteration from rsi. TÄÄ PITÄÄ PALAUTTAA JOKA SISÄLOOPISSA ALKAMAAN ALUSTA
    shr rcx,2 ;4 entries per loop. SISÄLOOPPI ON SE MISSÄ SSE vektorinopeutus käy
matmulInnerLoop:  ; w[] iterates only forward, add RDX, loop x from start on each inner loop.  Process
    movaps xmm1,[rdx] ;weight
    mulps xmm1,[r10] ; weightreg= weightreg*x
    haddps xmm1,xmm1
    haddps xmm1,xmm1 ; now This have scalar sum of 4 iterations
    addss xmm0,xmm1
    add rdx,16 ;NEXT 4 weight floats
    add r10,16; NEXT 4 X floats
    loop matmulInnerLoop
    
    movss [rdi],xmm0 ;one float to result
    add rdi,4 ; one float is 4 bytes!

    mov rcx,r8 ;use reg not stack. Ulkoloopin countteri takaisin
    loop matmulOuterLoop
    ret


;*****************
;**softmax, ********
;**************
;rdi:*x , rsi:size   (RBX, RBP, RSP and from R12 to R15 must be preserved by the called function)
softmax:
    mov rcx,rsi 
    mov rax,rdi ; do not miss *x
    movss xmm0,[rax]
    xorps xmm2,xmm2 ;Lets have sum here
softmaxMaxLoop:
    maxss xmm0,[rax]
    add rax,4; one 32bit float
    loop softmaxMaxLoop
    ;max_val is xmm0
    mov rcx,rsi
    mov rax,rdi ; re start x vector
softmaxSubLoop:
    movss xmm1,[rax]
    subss xmm1,xmm0
    movss [rax],xmm1
    add rax,4 
    loop softmaxSubLoop

    mov rcx,rsi
    mov rax,rdi
    push rsi
    push rdi
softmaxExpfLoop:
    movss xmm0, [rax]
    push rax
    push rcx
    call expf
    pop rcx
    pop rax
    movss [rax],xmm0
    add rax,4        
    loop softmaxExpfLoop
    pop rdi
    pop rsi

    ;Calculate sum into xmm1 register
    xorps xmm1,xmm1
    mov rcx,rsi
    mov rax,rdi
softmaxSumLoop:
    addss xmm1,[rax]
    add rax,4
    loop softmaxSumLoop
    ;now xmm1 have sum
    ;start normalization
    mov rcx,rsi
softmaxNormalizeLoop:
    movss xmm0,[rdi]
    divss xmm0,xmm1
    movss [rdi],xmm0 ;write back to mem
    add rdi,4
    loop softmaxNormalizeLoop
    ret

;***********************************
; argmax   rdi:*v, rsi:n   Return
sample_argmax: ; Loopataan maxps:llä läpi, kirjataan missä 4 floatin toistossa tuli ennätys ja xmm rekkari ylös  -> lopulta toisto*4+ solu xmm:ssä on vastaus
    mov rcx,rsi
    movss xmm0,[rdi]
    xor rax,rax
argmaxloop:
    movss xmm1,[rdi]
    comiss xmm1,xmm0
    jb argmaxNoUpdate
    mov rax,rsi
    sub rax,rcx
    movss xmm0,xmm1
argmaxNoUpdate:
    add rdi,4
    loop argmaxloop
ret

;***************************************
;***swigGLU(
;rdi:*x , rsi:*y, rdx:n,
;float *x,float *y,int n)     (RBX, RBP, RSP and from R12 to R15 must be preserved by the called function)
swigGLU:   ;x[i]= x[i]*(1 / (1 + expf(-x[i])))*y[i];
    push RBP
    mov RBP,RSP
    sub RSP,128

    ;OK instead of registers   swigGLU(st.hb,st.hb2,c->hidden_dim);
    mov rdi,[hb] ;RDI, RSI, RDX, RCX, R8
    mov rsi,[hb2]
    mov edx,[modelHeader+modelHeaderStruct.hiddenDim]
    mov rcx,rdx
swigGLUloop:
    ;calc exp(-x[i])
    xorps xmm0,xmm0  ;Zero
    movss xmm1, [rdi] ; X
    subss xmm0,xmm1  ;  0-x ,TODENNETTU OIKEAKS!
    push rsi
    push rdi
    push rcx
    call expf  ;PERKELE! Laskee vaan ydestä
    pop rcx
    pop rdi
    pop rsi
    
    movss xmm1,[rdi] ;x
    movss xmm2,[rsi] ;y
    

    ;OK  xmm0 have expf(-x)
    movss xmm1, [ones]
    addss xmm0,xmm1 ; 1+ expf(-x)
    rcpss xmm0,xmm0 ; 1/(1+ expf(-x))   ;TODO TARKKUUS!!!!
    ;EEII movaps [rdi],xmm0 ;JUST TEST

    mulss xmm0,[rdi] ; x* (1/(1+ expf(-x)))
    mulss xmm0,[rsi]
    movss [rdi],xmm0

    add rdi,4
    add rsi,4
    loop swigGLUloop

    
    add RSP,128
    pop RBP
    ret


;******************
;** updatefcrfci **
;****************** (RBX, RBP, RSP and from R12 to R15 must be preserved by the called function)
updatefcrfci: ;rdi: pos
    push RBP
    mov RBP,RSP
    sub RSP,128

    mov ecx, dword [modelHeader+modelHeaderStruct.dim]
    shr rcx,1 ;This is index for freqArr

    xorps xmm0,xmm0
    xorps xmm1,xmm1 ;easy debug?
    
    mov rax,[freqArr]
    mov rdx,[fcrfciArr] ;start from 0, each freq arr produces fcr fci entries
updatefcrfciloop:
    cvtsi2ss xmm1,rdi ;pos
    movss xmm0,[rax]
    mulss xmm0,xmm1 ; freq*pos

    push rax
    push rdx
    push rdi
    push rcx
    call cosf
    pop rcx
    pop rdi
    pop rdx
    pop rax

    movss [rdx],xmm0 ;cos
    add rdx,4

    cvtsi2ss xmm1,rdi ;pos
    movss xmm0,[rax]
    mulss xmm0,xmm1 ; freq*pos

    push rax
    push rdx
    push rdi
    push rcx
    call sinf
    pop rcx
    pop rdi
    pop rdx
    pop rax

    movss [rdx],xmm0 ;sin
    add rdx,4

    add rax,4;next freq entry
    loop updatefcrfciloop


    add RSP,128
    pop RBP
    ret

;************
;** rope ****
;************
; RDI:float *q,    RSI:float *k
rope:
    xor rcx,rcx
    mov ecx,[modelHeader+modelHeaderStruct.dim]
    shr rcx,1 ;how many times, not as index

    mov rdx,[fcrfciArr]
ROPEloop:
 
    movss xmm0,[rdi]   ;v0 = q[i];
    movss xmm1,[rdi]   ;v0 = q[i]
    movss xmm2,[rdi+4] ;v1 = q[i+1]
    movss xmm3,[rdi+4] ;v1 = q[i+1]

    mulss xmm0,[rdx]   ; v0 * fcrfciArr[i]
    mulss xmm2,[rdx+4] ; v1 * fcrfciArr[i+1]
    subss xmm0,xmm2    ; v0 * fcrfciArr[i] - v1 *fcrfciArr[i+1]
    movss [rdi],xmm0  ;st.q[i] 

    mulss xmm1,[rdx+4] ; v0 * fcrfciArr[i+1]
    mulss xmm3,[rdx]   ; v1 * fcrfciArr[i]
    addss xmm1,xmm3 ;
    movss [rdi+4],xmm1  ;st.q[i+1] = v0 * fci + v1 * fcr;

    movss xmm0,[rsi] ; v0 =  st.k[i]
    movss xmm1,[rsi] ; v0 =  st.k[i]
    movss xmm2,[rsi+4] ;v1 =  st.k[i+1]
    movss xmm3,[rsi+4] ;v1 = st.k[i+1]

    mulss xmm0,[rdx]  ; v0 * fcrfciArr[i]
    mulss xmm2,[rdx+4] ; v1 * fcrfciArr[i+1]
    subss xmm0,xmm2
    movss [rsi],xmm0  ;st.k[i] 

    mulss xmm1,[rdx+4]; v0 * fcrfciArr[i+1]
    mulss xmm3,[rdx]; v1 * fcrfciArr[i]
    addss xmm1,xmm3
    movss [rsi+4],xmm1  ;st.k[i+1] = v0 * fci + v1 * fcr;

    add rdi,8
    add rdx,8
    add rsi,8

    loop ROPEloop
    ret


;***********************************
;** forward( rdi:token, rsi:pos ), return logits **
;***********************************
forwardASM:
    push RBP
    mov RBP,RSP
    sub RSP,128
    ;memcpy(st.x, wei_token_embedding_table + token * c->dim, c->dim*sizeof(float));

    xor rdx,rdx
    mov [loff],rdx

    mov rax,rdi ;rdi needed once!
    mov ecx,[modelHeader+modelHeaderStruct.dim]
    mul rcx ; rax=rax*rcx
    shl rax,2 ; 4 bytes per float
    add rax,[wei_token_embedding_table]  ;this have source address
    mov rdi,[x] ;target

    shr rcx,1 ; 64bits have 2 floats    
forwardmemcpyloop:
    mov rdx,[rax]
    mov [rdi],rdx
    add rax,8 ;64bit
    add rdi,8 ;64bit
    loop forwardmemcpyloop

    ;reset pointers on forward method start

    mov rax,[wei_rms_att_weight]
    mov [point_rmsnorm],rax

    mov rax,[wei_wq]
    mov [point_wq],rax
    
    mov rax,[wei_wk]
    mov [point_wk],rax

    mov rax,[wei_wv]
    mov [point_wv],rax

    mov rax,[wei_wo]
    mov [point_wo],rax
    
    ;float *point_ffn_weight=wei_rms_ffn_weight;
    mov rax,[wei_rms_ffn_weight]
    mov [point_ffn_weight],rax
    
    mov rax,[wei_w1]
    mov [point_w1],rax

    mov rax,[wei_w2]
    mov [point_w2],rax

    mov rax,[wei_w3]
    mov [point_w3],rax

    ;re-set k and v states
    xor rdx,rdx
    mov rax,[kv_dim]
    mul rsi ;pos
    shl rax,2 ;4byte FLOAT 
    mov rdx,rax  ; rdx and rax have  pos * kv_dim*4;

    add rax,[key_cache]
    mov [k],rax ;st.k=st.key_cache + pos * kv_dim;
    add rdx,[value_cache]
    mov [v],rdx ;st.v=st.value_cache +  pos * kv_dim

    ;TODO NOW k and v are initialized TODO fix doROPE

    mov rdi,rsi ;pos as first argument
    push rsi
    call updatefcrfci
    pop rsi
    ;TODO loff alustus! ja seq_len add  seq_len*kv_dim  joka layer loopilla
    mov ecx,[modelHeader+modelHeaderStruct.n_layers]  ;initialize layer loop


;---finalize
    add RSP,128
    pop RBP
    ret

;*******************************
;****** weightedSumOfValues ****
;*******************************
;RDI:float *out,    RSI:float *v,  RDX:float *att,  RCX:int pos
weightedSumOfValues:
    push RBP
    mov RBP,RSP
    sub RSP,128

    push rcx
    mov rcx,[head_size]
    shr rcx,2; div by 4 when zeroing 128bit  4⁴4bytes in one loop like 
    push rdi
weightedSumOfValuesZeroingLoop: ;TODO loop like SSE SIMD
    mov qword [rdi],0
    add rdi,8
    mov qword [rdi],0
    add rdi,8
    loop weightedSumOfValuesZeroingLoop
    pop rdi
    ;----zeroing is done (todo check with gdb?)
    pop rcx ;get pos parametr back
    inc rcx  ; t<=pos
weightedSumOfValuesOUTERloop:
    push rcx ;store outer loop counter
    movss xmm0,[rdx] ;pick att[t]  this is "a"
    add rdx,4 ;next att[t] float "t" not needed
    push rsi  ;Store v pointer
    push rdi ;store out
    mov rcx,[head_size] ;init inner loop counter
weightedSumOfValuesINNERLoop:
    movss xmm1,[rsi]   ; v[i]
    add rsi,4; next float
    mulss xmm1,xmm0  ;xmm0 stays DO NOT CHANGE IN THIS LOOP  v[i]*a
    movss xmm2,[rdi]  ; xmm2=out[i]
    addss xmm2,xmm1   ; xmm2+=v[i]*a
    movss [rdi],xmm2 ;
    add rdi,4  ;just one float todo SIMD
    loop weightedSumOfValuesINNERLoop
    pop rdi ;get out[i] back where it was
    pop rsi ;get v[i] back where it was
    
    mov rcx,[kv_dim]
    shl rcx,2
    add rsi,rcx ;v+=kv_dim
     
    pop rcx
    loop weightedSumOfValuesOUTERloop

    add RSP,128
    pop RBP
    ret
;***********************************
;** multihead ********************************************************
;*(float* xb,float* q,float* att,float* key,float* kvalue,int pos){***
;RDI:*xb, RSI:*q, RDX:*att, RCX: *key, R8:*kvalue, R9:pos ************
;*********************************************************************
asmMultihead:
    push RBP
    mov RBP,RSP
    sub RSP,128

    ;float *point_q=st.q;  //Add head_size on every head
    ;float *point_att=st.att; //Add c->seq_len 
    ;float *point_k_layer_head=st.key_cache + loff; //add (c->dim*c->n_kv_heads) / (c->n_heads*c->n_heads),  on every
    ;float* point_xb = st.xb;

    mov [point_q],rsi
    mov [point_att],rdx
    mov [point_k_layer_head],rcx
    mov [point_xb],rdi

    ;float *point_layer_value=st.value_cache + loff;
    mov [point_layer_value],r8
    mov [var_pos],r9

    mov ecx,[modelHeader+modelHeaderStruct.n_heads]
asmHeadsLoop:
    push rcx
    mov rax,[point_k_layer_head]
    mov [point_k_layer_head_pos],rax

    mov rcx,[var_pos]
    inc rcx
    xor rax,rax
asmPosLoop:
    push rcx

    ;float score=dotproduct(point_q, point_k_layer_head_pos, head_size);
    ;RDI, RSI, RDX, RCX, R8, R9
    push rax
    mov rdi,[point_q]
    mov rsi, [point_k_layer_head_pos]
    mov rdx,[head_size]
    call dotproduct  ;RDI, RSI, RDX, RCX, R8, R9
    pop rax
    ;point_att[t] = score/sqrtf(head_size);
    divss xmm0,[sqrt_head_size]
    mov rdx,rax
    add rdx,[point_att]
    movss [rdx],xmm0
    ;point_k_layer_head_pos+=kv_dim;
    mov rcx,[kv_dim]
    shl rcx,2
    add [point_k_layer_head_pos],rcx
    add rax,4 ; t++  TODO TAVUJA VAI FLOATTEJA!!! 

    pop rcx
    loop asmPosLoop

    ;softmax(point_att, pos + 1);  ; point_att:RDI, pos+1:RSI
    mov rdi,[point_att]
    mov rsi,[var_pos] ;is only count?
    inc rsi
    call softmax

    ;RDI, RSI, RDX, RCX, R8, R9
    ;weightedSumOfValues(
    ;   point_xb,     ; RDI 
    ;   point_layer_value,  ; RSI
    ;   point_att,  ;RDX
    ;   pos)  ;RCX
    mov rdi,[point_xb]
    mov rsi,[point_layer_value]
    mov rdx,[point_att]
    mov rcx,[var_pos]
    call weightedSumOfValues

    ;point_layer_value+=kStep;
    ;point_k_layer_head+=kStep;
    mov rax,[kStep]
    shl rax,2
    add [point_layer_value],rax
    add [point_k_layer_head],rax

    ;point_q+=head_size;
    ;point_xb+=head_size;
    mov rax,[head_size]
    shl rax,2 ;TODO mul by 4 somewhere else
    add [point_q],rax
    add [point_xb],rax

    ;point_att+=c->seq_len;
    mov eax,[modelHeader+modelHeaderStruct.seq_len]
    shl rax,2
    add [point_att],rax


    pop rcx  ;too far for loop
    dec rcx
    jnz asmHeadsLoop

    add RSP,128
    pop RBP
    ret

;***********************************
;** doForward( rdi:token, rsi:pos ), return logits **
;***********************************
doForward:
    push RBP
    mov RBP,RSP
    sub RSP,128

    mov [var_pos],rsi  ; catch here before calls

    ;forwardASM(token,pos);
    call forwardASM ;same parameters, not needed anywhere else except pos from [var_pos]

    ;int loff=0;
    xor rax,rax
    mov [loff],rax
    
    ;float *point_rmsnorm=wei_rms_att_weight;
    mov rax,[wei_rms_att_weight]
    mov [point_rmsnorm],rax

    ;float *point_wq=wei_wq;
    mov rax,[wei_wq]
    mov [point_wq],rax
    ;float *point_wk=wei_wk; 
    mov rax,[wei_wk]
    mov [point_wk],rax
    ;float *point_wv=wei_wv;
    mov rax,[wei_wv]
    mov [point_wv],rax
    ;float *point_wo=wei_wo;
    mov rax,[wei_wo]
    mov [point_wo],rax

    ;float *point_ffn_weight=wei_rms_ffn_weight;
    mov rax,[wei_rms_ffn_weight]
    mov [point_ffn_weight],rax

    ;float *point_w1=wei_w1;
    mov rax,[wei_w1]
    mov [point_w1],rax
    ;float *point_w2=wei_w2;
    mov rax,[wei_w2]
    mov [point_w2],rax
    ;float *point_w3=wei_w3;
    mov rax,[wei_w3]
    mov [point_w3],rax

    ;st.k=st.key_cache + pos * kv_dim; //Add c->seq_len * kv_dim on every loop    HEEEII tää muuttaa caan C puolen pointteria!
    ;st.v=st.value_cache +  pos * kv_dim;

    xor rdx,rdx
    mov rax,[var_pos] ;pos goes here
    mul qword [kv_dim]
    shl rax,2 ;4bytes
    mov rcx,rax  ;pos*kv_dim  pos*288*4=1152
    add rax,[key_cache]
    mov [k],rax
    add rcx,[value_cache]
    mov [v],rcx

    ;updatefcrfci(pos);
    mov rdi,[var_pos]
    call updatefcrfci
 
    ;for(unsigned long long l = 0; l < c->n_layers; l++) {
    mov ecx,[modelHeader+modelHeaderStruct.n_layers]
startLayerLoop:
    push rcx
    ;attention rmsnorm
    ;rmsnorm(st.xb, st.x, point_rmsnorm, c->dim);      rdi:xb , rsi:*x, rdx:point_rmsnorm, rcx:dim
    mov rdi,[xb]
    mov rsi,[x]
    mov rdx,[point_rmsnorm]
    mov ecx,[modelHeader+modelHeaderStruct.dim]
    call rmsnorm

    ;qkv matmuls for this position
    ;matmul(st.q, st.xb, point_wq, c->dim, c->dim);   RDI, RSI, RDX, RCX, R8, R9
    mov rdi,[q]
    mov rsi,[xb]
    mov rdx,[point_wq]
    mov ecx,[modelHeader+modelHeaderStruct.dim]
    mov r8,rcx
    call matmul
    ;matmul(st.k, st.xb, point_wk, c->dim, kv_dim);
    mov rdi,[k]
    mov rsi,[xb]
    mov rdx,[point_wk]
    mov ecx,[modelHeader+modelHeaderStruct.dim]
    mov r8,[kv_dim]
    call matmul
    ;matmul(st.v, st.xb, point_wv, c->dim, kv_dim);
    mov rdi,[v]
    mov rsi,[xb]
    mov rdx,[point_wv]
    mov ecx,[modelHeader+modelHeaderStruct.dim]
    mov r8,[kv_dim]
    call matmul

    mov rdi,[q]
    mov rsi,[k]
    call rope ;rope(st.q,st.k);

    ;asmMultihead (st.xb,st.q,st.att,st.key_cache + loff,st.value_cache + loff,pos);
    ;RDI , RSI, RDX, RCX, R8
    mov rdi,[xb]
    mov rsi,[q]
    mov rdx,[att]
    mov rcx,[loff]
    add rcx,[key_cache]
    mov r8,[loff]
    add r8,[value_cache]
    mov r9,[var_pos]
    call asmMultihead

    ;final matmul to get the output of the attention
    ;matmul(st.xb2, st.xb, point_wo, c->dim, c->dim);  ;RDI , RSI, RDX, RCX, R8
    mov rdi,[xb2]
    mov rsi,[xb]
    mov rdx,[point_wo]
    mov ecx,[modelHeader+modelHeaderStruct.dim]
    mov r8,rcx
    call matmul

    ;residual connection back into x
    ;vecadd(st.x,st.xb2,c->dim);
    mov rdi,[x]
    mov rsi,[xb2]
    mov edx,[modelHeader+modelHeaderStruct.dim]
    call vecadd

    ;rmsnorm(st.xb, st.x, point_ffn_weight, c->dim);   ;RDI , RSI, RDX, RCX
    mov rdi,[xb]
    mov rsi,[x]
    mov rdx,[point_ffn_weight]
    mov ecx,[modelHeader+modelHeaderStruct.dim]
    call rmsnorm

    ;Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    ;first calculate self.w1(x) and self.w3(x)
    ;matmul(st.hb, st.xb, point_w1, c->dim, c->hidden_dim);  RDI, RSI, RDX, RCX, R8
    mov rdi,[hb]
    mov rsi,[xb]
    mov rdx,[point_w1]
    mov ecx,[modelHeader+modelHeaderStruct.dim]
    mov r8d,[modelHeader+modelHeaderStruct.hiddenDim]
    call matmul

    ;matmul(st.hb2, st.xb, point_w3, c->dim, c->hidden_dim);
    mov rdi,[hb2]
    mov rsi,[xb]
    mov rdx,[point_w3]
    mov ecx,[modelHeader+modelHeaderStruct.dim]
    mov r8d,[modelHeader+modelHeaderStruct.hiddenDim]
    call matmul

    ;// SwiGLU non-linearity
    ;swigGLU()
    call swigGLU

    ;// final matmul to get the output of the ffn
    ;matmul(st.xb, st.hb, point_w2, c->hidden_dim, c->dim);  ; RDI, RSI, RDX, RCX, R8
    mov rdi,[xb]
    mov rsi,[hb]
    mov rdx,[point_w2]
    mov ecx,[modelHeader+modelHeaderStruct.hiddenDim]
    mov r8d,[modelHeader+modelHeaderStruct.dim]
    call matmul

    ;vecadd(st.x,st.xb,c->dim);
    mov rdi,[x]
    mov rsi,[xb]
    mov edx,dword [modelHeader+modelHeaderStruct.dim]
    call vecadd


    ;point_rmsnorm+=c->dim;
    ;point_ffn_weight+=c->dim;
    mov eax,[modelHeader+modelHeaderStruct.dim]  ;tarkis  288
    shl rax,2 ;dim ei ollu kerrottu4:llä
    add [point_rmsnorm],rax
    add [point_ffn_weight],rax

    ;st.k+=c->seq_len * kv_dim;
    ;st.v+=c->seq_len * kv_dim;
    mov rax,[size_seqlen_kvDim] ; tarkis 294912 = 256*288*4 = 294912
    add [k],rax
    add [v],rax

    ;point_wq+=c->dim*c->dim;
    ;point_wo+=c->dim*c->dim;
    mov rax,[size_dim_dim]; tarkis 331776= 288*288*4 = 331776
    add [point_wq],rax
    add [point_wo],rax

    ;point_wk+=c->dim*kv_dim;
    ;point_wv+=c->dim*kv_dim;
    mov rax,[size_dim_kvDim]; tarkis 331776 = 288* 288*4
    add [point_wk],rax
    add [point_wv],rax


    ;point_w1+=c->dim*c->hidden_dim;
    ;point_w2+=c->dim*c->hidden_dim;        
    ;point_w3+=c->dim*c->hidden_dim;
    mov rax,[size_dim_hiddenDim]  ;884736 = 288*768*4 = 884736
    add [point_w1],rax
    add [point_w2],rax
    add [point_w3],rax
    
    ;loff += c->seq_len * kv_dim;
    mov rax,[size_seqlen_kvDim]  ;294912 = 256*288*4 = 294912 
    add [loff],rax

    pop rcx ;too far for loop
    dec rcx
    jnz startLayerLoop
    ;loop startLayerLoop

    ;rmsnorm(st.x, st.x, wei_rms_final_weight, c->dim);
    mov rdi,[x]
    mov rsi,[x]
    mov rdx,[wei_rms_final_weight]
    mov ecx,[modelHeader+modelHeaderStruct.dim]
    call rmsnorm
    ;classifier into logits
    ;matmul(st.logits, st.x, wei_wcls, c->dim, c->vocab_size);  RDI, RSI, RDX, RCX, R8
    mov rdi,[logits]
    mov rsi,[x]
    mov rdx,[wei_wcls]
    mov ecx,[modelHeader+modelHeaderStruct.dim]
    mov r8d,[modelHeader+modelHeaderStruct.vocabSize]
    call matmul   
    ;return st.logits;
    mov rax,[logits]

;---finalize do layers loop
    add RSP,128
    pop RBP
    ret


;*****************
;* ReadPrompt ****
;*****************
;rdi: *target
readPrompt:
    xor rax,rax
    push rdi
    mov rdi,0   ; stdin 0
    mov rsi,promptext
    mov rdx,1024
    syscall
    pop rdi
    mov rcx,promptext
    xor rdx,rdx ;add termination \0 to \n
    dec rax
    mov [rcx+rax],rdx
    mov [rdi],rcx
    ret
;*****************
; PrintTokenText**
;*****************
;rdi: token
printTokenText:
    ;set buffer
    mov rsi,rdi
    shl rsi,5 ;32bytes
    add rsi, [tokenTextsAddr]

    xor rdx,rdx ;count bytes
printTokenTextCountBytes:
    inc rdx ; not zero len token :)
    mov al,byte [rsi+rdx]
    cmp al,0
    jnz printTokenTextCountBytes

    mov rax,1  ; sys_write?  
    mov rdi,1 ; 
    syscall
    ret
;*****************
; Start timing ***
;*****************
getClock:
    mov rdi, 1                    ; CLOCK_MONOTONIC (1)
    mov rsi, startTime_sec
    mov rax, 0xE4                 ; syscall number for clock_gettime (E4 in hexadecimal)
    syscall

    ; Convert seconds and nanoseconds to milliseconds
    ; milliseconds = tv_sec * 1000 + tv_nsec / 1000000
    mov rax, [startTime_sec] ; tv_sec
    ; Convert seconds to milliseconds
    mov rbx, 1000
    mul rbx                        ; rax = tv_sec * 1000
    mov rcx,rax ; rcx have sec part in milliseconds

    mov rax, [startTime_nsec] ; tv_nsec
    xor rdx,rdx
    mov rbx,1000000
    div rbx
    add rax, rcx                   ; rax now contains the time in milliseconds
    ret



;***********
;** main ***
;***********

_start:
main:
    push RBP
    mov RBP,RSP
    sub RSP,128
    push RBX
    push R12
    push R13
    push R14
    push R15
    ;----- load  todo change name to better
    call loadAI ; No parameters just run
    ;--read prompt from input
    xor rax,rax
    mov rdi,0   ; stdin 0
    mov rsi,promptext
    mov rdx,1024
    syscall
    mov rcx,promptext
    xor rdx,rdx ;add termination \0 to \n
    dec rax
    mov [rcx+rax],rdx
    ;-------------------------    
    ;rdi: *text , rsi: *tokens, rdx: *n_tokens
    mov rdi,promptext
    mov rsi,prompttokens
    mov rdx,promptokencount
    call encode
    ;----- Feed prompt -------
    xor rsi,rsi
    xor rdi,rdi
    mov edi,dword[prompttokens]
    mov rcx,[promptokencount]
    dec rcx
    jz predict

feedpromptloop:

    push rcx
    push rsi
    call doForward ;( rdi:token, rsi:pos ), return logits **
    pop rsi
    pop rcx

    inc rsi
    xor rdi,rdi
    mov edi,dword[prompttokens+rsi*4]

    push rcx
    push rsi
    push rdi
    call printTokenText
    pop rdi
    pop rsi
    pop rcx
    loop feedpromptloop

    ;lets have honest timing, record from staring of predicting new tokens that include argmax

    ;---- Predict part -------
predict:
    push rsi
    push rdi
    call getClock
    mov [clockvalue],rax
    pop rdi
    pop rsi


    mov rcx,256
    sub rcx,[promptokencount]
    ;rdi is token, comes first at prev loop. Later for token picked by logits
    ;rsi is position, continue
predictloop:
    push rcx
    push rsi
    push rdi
    call printTokenText
    pop rdi
    pop rsi
    pop rcx

    push rcx
    push rsi ;only pos needed to be stored
    call doForward ;( rdi:token, rsi:pos ), return logits **

    mov rdi,rax
    xor rsi,rsi
    mov esi,[modelHeader+modelHeaderStruct.vocabSize]
    call sample_argmax ; (logits, c->vocab_size);
    mov rdi,rax ;next token
    pop rsi
    pop rcx
    inc rsi

    cmp rax,1
    je endpredict

    loop predictloop

endpredict:
    ;---Report tok per sec
    ;fmtTextTokensPerSec
    push rsi
    call getClock
    pop rsi
    sub rax,[clockvalue]
    sub rsi,[promptokencount]
    mov rbx,rax
    mov rax,rsi
    mov rsi,rbx  ;now clocks in rsi, tokens in rax
    
    mov rbx,1000 ;it is better to multiply with 1000 than div by 1000 when precision is lost
    mul rbx
    div rsi

    mov rsi,rax
    xor rax,rax
    mov rdi,fmtTextTokensPerSec
    call printf

    pop R15
    pop R14    
    pop R13
    pop R12
    pop RBX
    add RSP,128
    pop RBP
    ret

