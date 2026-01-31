<button 
  className={styles.collapseBtn}
  type="button" 
  aria-expanded={isExpanded}
  title="Collapse layer"
>
  {layerIndex < 2 ? (isExpanded ? '▼' : '▶') : '▶'}
</button>