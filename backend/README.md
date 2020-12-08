## Query spec

1. Getter and Setter of model
- Backend Response (GET)
```
{
    model_name: "betaB",
    tsne_length: t,
    vis_B_shape: [a,b],
    vis_C_length: c,
    delta: d
}
```
- Frontend Query (POST)
```
{
    opcode: 'set_param',
    content: [
        {
            param_name: "delta",
            value: d
        }
    ]
}
```

2. TSNE Visualization
    - x,y,~ : Real number
    - z : Integer
- Frontend Query (POST)
```
{
    opcode: 'tsne',
    content: []
}
```
- Backend Response
```
{
    opcode: 'tsne',
    content: [
        {
            latent: [~,...,~],
            img: (base64),
            tsne_pos: [x,y],
            label: z
        },
        , ...
    ]
}
```

3. Latent Visualization
    - ~ : Real number
    - a, b : Integer (index of dimension)
- Frontend Query (POST)
```
{
    opcode: 'latent_imgs'
    content: [
        {
            latent: [~,...,~],
            target_idx: [a,b]
        }
    ]
}
```
- Backend Response
```
{
    opcode: 'latent_imgs',
    content: {
        target: {
            latent: [~,...,~],
            img: (base64)
        },
        tile: [
            {
                latent: [~,...,~],
                img: (base64)
            }, ...
        ],
        linear: [
            {
                latent: [~,...,~],
                img: (base64)
            }, ...
        ]
    }
}
```


4. Get min and max of each dimension
    - ~ : Real number
- Frontend Query (POST)
```
{
    opcode: 'min_max',
    content: []
}
```
- Backend Response
```
{
    opcode: 'min_max',
    content: [
        {
            'min': [~,...,~],
            'max': [~,...,~]
        }
    ]
}
```
