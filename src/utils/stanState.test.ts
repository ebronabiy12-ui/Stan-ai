import test from 'node:test';
import assert from 'node:assert/strict';
import {
  clampGenerationSettings,
  computeLossDomain,
  formatParamCount,
  formatTrainingStatusLabel,
  isProductionChatReady,
} from './stanState';

test('isProductionChatReady strictly gates untrained and smoke_test states', () => {
  assert.equal(
    isProductionChatReady({
      checkpoint_loaded: false,
      tokenizer_loaded: false,
      training_status: 'untrained',
    }),
    false,
  );
  assert.equal(
    isProductionChatReady({
      checkpoint_loaded: true,
      tokenizer_loaded: true,
      training_status: 'smoke_test',
    }),
    false,
  );
  assert.equal(
    isProductionChatReady({
      checkpoint_loaded: true,
      tokenizer_loaded: false,
      training_status: 'trained',
    }),
    false,
  );
  assert.equal(
    isProductionChatReady({
      checkpoint_loaded: true,
      tokenizer_loaded: true,
      training_status: 'trained',
    }),
    true,
  );
  assert.equal(
    isProductionChatReady({
      checkpoint_loaded: true,
      tokenizer_loaded: true,
      training_status: 'instruction-tuned',
    }),
    true,
  );
});

test('clampGenerationSettings bounds all parameters cleanly', () => {
  const clamped = clampGenerationSettings({
    temperature: 5.5,
    top_k: -10,
    top_p: 0,
    max_new_tokens: 9999,
    repetition_penalty: 0.4,
    seed: -3,
    apiBaseUrl: 'http://127.0.0.1:8000/',
  });
  assert.equal(clamped.temperature, 2);
  assert.equal(clamped.top_k, 0);
  assert.equal(clamped.top_p, 0.05);
  assert.equal(clamped.max_new_tokens, 512);
  assert.equal(clamped.repetition_penalty, 1);
  assert.equal(clamped.seed, null);
  assert.equal(clamped.apiBaseUrl, 'http://127.0.0.1:8000');
});

test('formatParamCount and status labels render accurate human-readable values', () => {
  assert.equal(formatParamCount(51658240), '51,658,240 (51.66M)');
  assert.equal(formatTrainingStatusLabel('untrained'), 'Untrained (No Checkpoint)');
  assert.equal(formatTrainingStatusLabel('instruction-tuned'), 'Instruction-Tuned');
});

test('computeLossDomain extracts valid curves and handles empty histories', () => {
  const empty = computeLossDomain([], 'train_loss');
  assert.equal(empty.points.length, 0);

  const domain = computeLossDomain(
    [
      { step: 10, train_loss: 4.0, val_loss: null, learning_rate: 3e-4 },
      { step: 20, train_loss: 2.5, val_loss: 2.8, learning_rate: 2e-4 },
    ],
    'train_loss',
  );
  assert.equal(domain.points.length, 2);
  assert.equal(domain.minStep, 10);
  assert.equal(domain.maxStep, 20);
  assert.ok(domain.minVal < 2.5);
  assert.ok(domain.maxVal > 4.0);
});
