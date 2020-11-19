# InfoVis20202
For InfoVis 2020 final project

### Query spec

1. TSNE Visualization
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

2. Latent Visualization
    - ~ : Real number
    - a, b : Integer (index of dimension)
- Frontend Query (POST)
```
{
	opcode: 'latent_imgs'
	content: {
		latent: [~,...,~],
		target_idx: [a,b]
	}
}
```
- Backend Response
```
{
	opcode: 'latent_imgs',
	content: [
		{
			latent: [~,...,~],
			img: (base64)
		}, ...
	]
}
```
