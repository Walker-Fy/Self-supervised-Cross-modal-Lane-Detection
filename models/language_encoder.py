"""Language (text) encoder using CLIP."""

from typing import List, Union, Dict

import torch
import torch.nn as nn


class LanguageEncoder(nn.Module):
    """Text encoder using CLIP's text encoder.

    Frozen by default as specified in SCC paper.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        frozen: bool = True,
        max_length: int = 77,
    ):
        """Initialize language encoder.

        Args:
            model_name: CLIP model name (must match visual encoder)
            pretrained: Pretrained weights
            frozen: Whether to freeze the encoder
            max_length: Maximum text sequence length
        """
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.frozen = frozen
        self.max_length = max_length

        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch is required. Install with: pip install open_clip_torch"
            )

        # Load CLIP model
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

        # Extract text encoder
        self.text_encoder = self.clip_model.text

        # Get output dimension
        self.output_dim = self.text_encoder.output_dim

        # Get tokenizer
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Freeze if specified
        if frozen:
            self._freeze()

    def _freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self.frozen = False

    def tokenize(
        self,
        texts: Union[str, List[str]],
    ) -> torch.Tensor:
        """Tokenize input text(s).

        Args:
            texts: Input text or list of texts

        Returns:
            Tokenized tensor (B, max_length)
        """
        return self.tokenizer(texts)

    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """Encode text(s) to embeddings.

        Args:
            texts: Input text or list of texts
            normalize: Whether to L2 normalize output

        Returns:
            Text embeddings (B, output_dim)
        """
        tokens = self.tokenize(texts)

        # Move to same device as model
        if hasattr(self, "device"):
            tokens = tokens.to(self.device)
        else:
            device = next(self.parameters()).device
            tokens = tokens.to(device)

        return self.forward(tokens, normalize=normalize)

    def forward(
        self,
        tokens: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: Tokenized text (B, max_length)
            normalize: Whether to L2 normalize output

        Returns:
            Text embeddings (B, output_dim)
        """
        # Encode with CLIP text encoder
        x = self.text_encoder(tokens)

        # Take features from eot token (end of text)
        if x.dim() > 2:
            x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]

        if normalize:
            x = F.normalize(x, dim=-1)

        return x

    def get_text_embeddings(
        self,
        prompts: List[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """Get embeddings for a list of prompts.

        Useful for creating a fixed vocabulary of lane descriptions.

        Args:
            prompts: List of text prompts
            normalize: Whether to L2 normalize

        Returns:
            Text embeddings (len(prompts), output_dim)
        """
        return self.encode_text(prompts, normalize=normalize)


class TextPromptGenerator(nn.Module):
    """Generate and encode text prompts for lane detection.

    Creates a vocabulary of lane descriptions and caches their embeddings.
    """

    def __init__(
        self,
        language_encoder: LanguageEncoder,
        max_lanes: int = 6,
    ):
        """Initialize prompt generator.

        Args:
            language_encoder: Text encoder to use
            max_lanes: Maximum number of lanes for prompt generation
        """
        super().__init__()

        self.encoder = language_encoder
        self.max_lanes = max_lanes

        # Generate all prompts
        self.prompts = self._generate_prompts()
        self.register_buffer("cached_embeddings", None)

    def _generate_prompts(self) -> List[str]:
        """Generate all possible lane description prompts.

        Returns:
            List of prompt strings
        """
        prompts = []

        lane_counts = list(range(2, self.max_lanes + 1))
        line_types = ["solid", "dashed", "double", "mixed"]

        for count in lane_counts:
            for line_type in line_types:
                prompts.append(f"a {count}-lane road with {line_type} center line")
                prompts.append(f"a road with {count} lanes and {line_type} markings")

        return prompts

    def get_cached_embeddings(self) -> torch.Tensor:
        """Get cached prompt embeddings.

        Returns:
            Cached embeddings (num_prompts, output_dim)
        """
        if self.cached_embeddings is None:
            self.cached_embeddings = self.encoder.encode_text(self.prompts)
        return self.cached_embeddings

    def forward(self) -> torch.Tensor:
        """Get all prompt embeddings.

        Returns:
            Prompt embeddings (num_prompts, output_dim)
        """
        return self.get_cached_embeddings()

    def find_best_match(
        self,
        visual_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> tuple:
        """Find best matching prompt for visual embedding.

        Args:
            visual_embedding: Visual feature (B, output_dim)
            top_k: Number of top matches to return

        Returns:
            Tuple of (top_indices, top_scores, top_prompts)
        """
        prompt_embeddings = self.get_cached_embeddings()

        # Compute similarity
        similarity = (visual_embedding @ prompt_embeddings.T)

        # Get top k
        top_scores, top_indices = similarity.topk(top_k, dim=-1)

        top_prompts = [[self.prompts[i] for i in indices] for indices in top_indices]

        return top_indices, top_scores, top_prompts
