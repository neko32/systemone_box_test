# API 仕様

neko-box はサーバを持たない CLI。ここでは CLI のインターフェースと、利用する外部 API（TypeSafe System One）の呼び出し仕様をまとめる。

## CLI: `neko-box`

```
uv run neko-box [--input PATH] [--output PATH] [--mode live|offline] [--interval SEC]
                [--prompt TEXT] [--max-cats N] [--threshold P] [--no-display] [-v]
```

| オプション | 既定値 | 説明 |
|---|---|---|
| `--input` | `movie/neko.mp4` | 入力動画 |
| `--output` | `out/neko_boxed.mp4` | 注釈付きの出力動画（mp4v） |
| `--mode` | `live` | `live`: 再生しながら非同期検出 / `offline`: 同期検出して書き出し、その後再生 |
| `--interval` | `1.0` | スナップショットの間隔（秒） |
| `--prompt` | `猫はどこですか` | Box の instructions（1 匹モードのみ） |
| `--max-cats` | `1` | 1 スナップショットで探す猫の最大数（1〜10）。2 以上で複数匹モード（猫の数だけ順番にリクエストする） |
| `--threshold` | `0.5` | 猫がいると判定する Noul 確率の下限 |
| `--no-display` | off | ウィンドウを出さない（`offline` のみ） |
| `-v` | off | DEBUG ログ（presence / confidence / 座標） |

環境変数: `API_KEY_JEV`（必須）。`.env` からも読み込む。
ウィンドウでは `q` または `ESC` で終了する。

## 外部 API: TypeSafe System One

- `POST https://api.typesafe.ai/v1/systemone`（SDK: `typesafe-sdk` の `TypeSafeClient.system_one`）
- ヘッダ: `Authorization: Bearer <API_KEY_JEV>`、`TypeSafe-Danger-Zone-Images: true`
- モデル: `jev-preview`（現在は `jev-1.14.0`）

### リクエスト（SDK が multipart に変換する）
```json
{
  "model": "jev-preview",
  "state": "<image:...>",
  "assets": {"<image:...>": {"type": "image", "part": "..."}},
  "questions": {
    "present": {"type": "noul", "instructions": "この画像に猫は写っていますか？"},
    "cat":     {"type": "box",  "instructions": "猫はどこですか"}
  }
}
```
画像: JPEG（品質 90）、各辺 280〜1568px。

複数匹モード（`--max-cats N`、N ≥ 2）では次のように聞く。
- 1 回目：`present_1`..`present_N` に Noul「この画像に猫が少なくとも{k}匹写っていますか？」、`cat` に Box「1匹の猫」
- 2 回目以降（確定した匹数まで）：見つけた猫をグレーで塗りつぶした画像について、`cat` に Box「1匹の猫」だけ

### レスポンス（実測例）
```json
{
  "model": "jev-1.14.0",
  "answers": {
    "present": {"type": "noul", "noul": 0.96},
    "cat": {"type": "box",
            "box": {"top": 0.23, "bottom": 0.89, "left": 0.57, "right": 0.77},
            "confidence": 0.21}
  },
  "usage": {"input_tokens": 2721, "output_tokens": 36}
}
```
`box` は左上原点の正規化座標（0〜1）。ピクセル座標 = `left*W, top*H, right*W, bottom*H`。

### エラー処理
| ステータス | 扱い |
|---|---|
| 400 / 401 / 413 / 422 | `DetectionError` → そのスナップショットは「猫なし」、警告ログ |
| 429 / 529 / 5xx | SDK が指数バックオフでリトライ → それでも失敗したら上と同じ |
