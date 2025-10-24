# Document Title

```@setup experiment1
using Bonito
Bonito.Page()
```


```@example experiment1
using RunwayLib
using Bonito, BonitoBook
App() do
    path = normpath(joinpath(dirname(pathof(RunwayLib)), "..", "docs", "src", "test.md"))
    BonitoBook.InlineBook(path)
end
```

